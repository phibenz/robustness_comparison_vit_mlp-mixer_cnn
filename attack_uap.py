import argparse, yaml, os, sys
import shutil
from contextlib import suppress
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision.utils as vutils

from models.uap import UAP
from utils import Normalize, Unnormalize, get_logger, get_timestamp, load_ground_truth, get_model
from config import IMAGENET_PATH, NEURIPS_DATA_PATH, NEURIPS_CSV_PATH

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='UAP Attack')
# Dataset / Model parameters
parser.add_argument('--source-model', nargs="+", default=['resnet101'], help='Source model')
parser.add_argument('--target-model', nargs="+", default=['resnet101'], help='Target model')         
parser.add_argument('--dataset', default='imagenet', type=str, help='Used Dataset')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

# Universal Adversarial Attack
parser.add_argument('--attack-iterations', type=int, default=2000, help='Number of attack iterations')
parser.add_argument('--attack-loss-fn', default='ce-untargeted', help='Adversarial attack loss function')
parser.add_argument('--attack-lr', type=eval, default=0.005, help='Attack step size')
parser.add_argument('--attack-epsilon', type=eval, default=16/255, help='Epsilon')
parser.add_argument('--attack-class', default=0, type=int, help='Adversarial attack variant')                                      

# Misc
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--workers', type=int, default=8, help='Data loading workers')
parser.add_argument('--subfolder', type=str, default='', help='Subfolder to store the results in')
parser.add_argument('--postfix', type=str, default='', help='Postfix to append to results folder')

class NeurIPSLoader(Dataset):
    def __init__(self, data_path, csv_path, transform=None, sampling_frequency=1):
        # Data type handling must be done beforehand. It is too difficult at this point.
        self.image_id_list, self.label_ori_list, self.label_tar_list = load_ground_truth(os.path.join(csv_path, 'images.csv'))
        self.image_id_list = [x + '.png' for x in self.image_id_list]
        self.image_id_list = self.image_id_list[0 : 1000 : sampling_frequency]
        self.label_ori_list = self.label_ori_list[0 : 1000 : sampling_frequency]
        self.label_tar_list = self.label_tar_list[0 : 1000 : sampling_frequency]
        self.data_path = data_path
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.data_path + self.image_id_list[index])
        if self.transform:
            x = self.transform(x)
        y_gt= self.label_ori_list[index]
        y_tar = self.label_tar_list[index]
        return x, (y_gt, y_tar)

    def __len__(self):
        return len(self.image_id_list)

def _parse_args():
    args = parser.parse_args()
    return args

def main():
    args = _parse_args()

    args.distributed = False

    args.device = 'cuda:0'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    result_path = os.path.join('./output', 'attack', args.subfolder, get_timestamp() + args.postfix)
    os.makedirs(result_path)

    # Saving this file
    shutil.copy(sys.argv[0], os.path.join(result_path, sys.argv[0]))
    _logger = get_logger(result_path)

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        _logger.info('{} : {}'.format(key, value))

    amp_autocast = suppress  # do nothing

    if args.dataset == 'imagenet':
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_eval = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])

        dir_eval = os.path.join(IMAGENET_PATH, 'val')
        data_eval = dset.ImageFolder(root=dir_eval, transform=transform_eval)

        if args.attack_targeted:
            num_samples=len(data_eval.targets)
            rnd=np.random.randint(1, num_classes,(num_samples,))
            data_eval.targets=(data_eval.targets+rnd)%num_classes

        loader_eval = torch.utils.data.DataLoader(data_eval,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    elif args.dataset == 'neurips':
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_eval = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
        data_eval = NeurIPSLoader(NEURIPS_DATA_PATH, NEURIPS_CSV_PATH, transform_eval, sampling_frequency=1)
        loader_eval = torch.utils.data.DataLoader(data_eval,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    eval_iter=iter(loader_eval)

    # UAP
    uap = UAP(shape=(224, 224), num_channels=3).cuda()
    optimizer = torch.optim.Adam(uap.parameters(), lr=args.attack_lr)
    
    source_model=[]
    for sm in args.source_model:
        model=get_model(sm)
        model.cuda()
        source_model.append(model)  

    target_model=[]
    for tm in args.target_model:
        model=get_model(tm)
        model.cuda()
        target_model.append(model)
    num_target_models=len(args.target_model)

    ################# ATTACK ######################
    unnorm = Unnormalize(mean=mean, std=std)
    norm = Normalize(mean=mean, std=std)
    norm_vit = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    num_samples = 0
    for it in range(args.attack_iterations):
        if it%20 == 0:
            print('{} / {}'.format(it+1, args.attack_iterations))
        try: 
            x, lbl = next(eval_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            eval_iter = iter(loader_eval)
            x, lbl = next(eval_iter)

        if args.dataset=='neurips':
            y_gt=lbl[0]
        elif args.dataset=='imagenet':
            y_gt=lbl 
        y_tar = torch.ones_like(y_gt)*args.attack_class

        x = x.cuda()
        y_gt = y_gt.cuda()
        y_tar = y_tar.cuda()

        x_unnorm = unnorm(x)
        x_adv = uap(x_unnorm)

        logits=0
        for sm, sm_name in zip(source_model, args.source_model):
            with amp_autocast():
                if 'ViT' in sm_name:
                    lo = sm(norm_vit(x_adv))
                elif 'mixer_' in sm_name:
                    lo = sm(norm_vit(x_adv))
                else:
                    lo = sm(norm(x_adv))
            if isinstance(lo, (tuple, list)):
                lo = lo[0]
            logits += lo

        if args.attack_loss_fn == "ce-untargeted":
            loss = -nn.CrossEntropyLoss().cuda()(logits, y_gt)
        elif args.attack_loss_fn=='ce-targeted':
            loss = nn.CrossEntropyLoss().cuda()(logits, y_tar)
        else:
            raise ValueError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        uap.delta.data = torch.clamp(uap.delta.data, -args.attack_epsilon, args.attack_epsilon)

    torch.save(uap.delta.data, os.path.join(result_path, 'uap.pth'))
    # Store results
    acc_untargeted=np.zeros((num_target_models))
    acc_targeted=np.zeros((num_target_models))
    num_samples=0

    for x, lbl in loader_eval:        
        if args.dataset=='neurips':
            y_gt=lbl[0]
        elif args.dataset=='imagenet':
            y_gt=lbl

        y_tar = torch.ones_like(y_gt)*args.attack_class
        x = x.cuda()
        y_gt = y_gt.cuda()
        y_tar = y_tar.cuda()
        
        x_unnorm = unnorm(x)
        x_adv = uap(x_unnorm)
        for tm_idx, (tm, tm_name) in enumerate(zip(target_model, args.target_model)):
            if 'ViT' in tm_name:
                lo = tm(norm_vit(x_adv))[0]
            elif 'mixer_' in tm_name:
                lo = tm(norm_vit(x_adv))[0]                
            else:
                lo = tm(norm(x_adv))
            pred = torch.argmax(lo, dim=-1)
            # Get the number of correctly classified samples
            corr_cl = sum(pred == y_gt).cpu().numpy()
            acc_untargeted[tm_idx] += corr_cl
            # Get the number of correctly targeted samples
            corr_tar = sum(pred == y_tar).cpu().numpy()
            acc_targeted[tm_idx] += corr_tar
        
        num_samples+=len(x)

    # Results
    _logger.info('\n-- Untargeted ASR --')
    for tm_idx, tm in enumerate(args.target_model):  
        _logger.info('{} -> {}'.format(args.source_model, tm))
        _logger.info('{}'.format(1.-acc_untargeted[tm_idx]/num_samples))

    _logger.info('\n-- Targeted Accuracy --')
    for tm_idx, tm in enumerate(args.target_model):   
        _logger.info('{} -> {}'.format(args.source_model, tm))
        _logger.info('{}'.format(acc_targeted[tm_idx]/num_samples))  

    model_string=''
    untargeted_string=''
    targeted_string=''
    untar_tar_string=''
    for tm_idx, tm in enumerate(args.target_model):     
        model_string += tm + ' '
        untargeted_string += '{} '.format(1.-acc_untargeted[tm_idx]/num_samples)
        targeted_string += '{} '.format(acc_targeted[tm_idx]/num_samples)
        untar_tar_string += '{}/{} '.format(1.-acc_untargeted[tm_idx]/num_samples,acc_targeted[tm_idx]/num_samples)

    _logger.info('{}'.format(model_string))
    _logger.info('-- Untargeted ASR --')
    _logger.info(untargeted_string)
    _logger.info('-- Targeted Accuracy --')
    _logger.info(targeted_string)
    _logger.info('-- Untargeted ASR / Targeted Accuracy --')
    _logger.info(untar_tar_string)
    
    # Saving the UAP
    uap_viz = uap.delta.data.cpu().clone()
    uap_viz = uap_viz - uap_viz.min()
    uap_viz = uap_viz / uap_viz.max()
    vutils.save_image(uap_viz, os.path.join(result_path, 'uap.png'))

if __name__ == '__main__':
    main()
