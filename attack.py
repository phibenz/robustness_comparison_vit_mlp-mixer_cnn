import argparse, os, sys
import shutil
from contextlib import suppress

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import Dataset

from utils import Normalize, Unnormalize, get_logger, get_timestamp, load_ground_truth, get_model
from config import IMAGENET_PATH, NEURIPS_DATA_PATH, NEURIPS_CSV_PATH

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Adversarial Attack')
# Dataset / Model parameters
parser.add_argument('--source-model', nargs="+", default=['resnet101'], help='Name of model to train')
parser.add_argument('--target-model', nargs="+", default=['resnet101'], help='Name of model to train')
parser.add_argument('--dataset', default='imagenet', type=str, help='Used Dataset')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

# Adversarial Attack
parser.add_argument('--attack-iterations', type=int, default=7, help='Number of attack iterations')
parser.add_argument('--attack-init', type=str, default='zeros', help='Initialization of the perturbation')
parser.add_argument('--attack-loss-fn', default='ce-untargeted', help='Adversarial attack loss function')
parser.add_argument('--attack-step-size', type=eval, default=2./255., help='Attack step size')
parser.add_argument('--attack-epsilon', type=eval, default=16/255, help='Epsilon')
parser.add_argument('--attack-log-interval', type=int, default=1, help='Log interval')
parser.add_argument('--attack-norm', type=str, default='linf', help='Norm')

# Misc
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--log-interval', type=int, default=50, help='logging interval')
parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
parser.add_argument('--subfolder', default='', type=str, help='Subfolder to store the results in')
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
    assert args.attack_iterations % args.attack_log_interval == 0
    return args

def main():
    args = _parse_args()
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

        loader_eval = torch.utils.data.DataLoader(data_eval,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
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

    norm = Normalize(mean=mean, std=std)
    norm_vit = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    unnorm = Unnormalize(mean=mean, std=std)

    ################# ATTACK ######################

    # Store results
    acc_untargeted=np.zeros((num_target_models, args.attack_iterations // args.attack_log_interval))
    acc_targeted=np.zeros((num_target_models, args.attack_iterations // args.attack_log_interval))
    norm_l2 = np.zeros((num_target_models, args.attack_iterations // args.attack_log_interval))
    norm_linf = np.zeros((num_target_models, args.attack_iterations // args.attack_log_interval))
    num_samples = 0

    last_idx = len(loader_eval) - 1

    for batch_idx, (x, lbl) in enumerate(loader_eval):
        print("Attack: {}/{}".format(batch_idx, last_idx))
        if args.dataset == 'neurips':
            y_gt = lbl[0]
            y_tar = lbl[1]
        elif args.dataset == 'imagenet':
            y_gt = lbl 
            rnd = torch.randint(1, num_classes, (len(lbl),))
            y_tar = (y_gt + rnd) % num_classes
        
        x = x.cuda()
        y_gt = y_gt.cuda()
        y_tar = y_tar.cuda()

        # Initialize delta
        if args.attack_init == 'zeros':
            delta = torch.zeros_like(x, requires_grad=True)
        elif args.attack_init == 'gauss':
            delta = torch.randn_like(x, requires_grad=True) * args.attack_epsilon
        elif args.attack_init == 'uniform':
            noise = torch.rand_like(x) * 2 * args.attack_epsilon - args.attack_epsilon
            delta = noise.clone().detach().requires_grad_(True)
        else:
            raise ValueError
        
        delta = torch.zeros_like(x, requires_grad=True)

        for t in range(args.attack_iterations):
            # Add the perturbation to the unnormalized image
            x_unnorm = unnorm(x)
            x_adv = x_unnorm + delta

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

            loss.backward()

            grad_a = delta.grad.clone()
            delta.grad.zero_()
            delta.data = delta.data - args.attack_step_size * torch.sign(grad_a)

            if args.attack_norm == 'linf':
                delta.data = delta.data.clamp(-args.attack_epsilon, args.attack_epsilon)
            elif args.attack_norm == 'l2':
                # Get the norm of each sample
                shape = x.size()
                pixel_num = shape[1]*shape[2]*shape[3]
                magnitude = ((pixel_num * (args.epsilon)**2)**0.5)
                delta.data = delta.data.renorm(p=2, dim=0, maxnorm=magnitude)
            else:
                raise ValueError

            delta.data = ((x_unnorm + delta.data).clamp(0, 1)) - x_unnorm

            if t % args.attack_log_interval == (args.attack_log_interval - 1):
                for tm_idx, (tm, tm_name) in enumerate(zip(target_model, args.target_model)):
                    if 'ViT' in tm_name:
                        lo = tm(norm_vit(x_unnorm + delta))
                    elif 'mixer_' in tm_name:
                        lo = tm(norm_vit(x_unnorm + delta))                        
                    else:
                        lo = tm(norm(x_unnorm + delta))
                    if isinstance(lo, (tuple, list)):
                        lo = lo[0]
                    pred = torch.argmax(lo, dim=-1)
                    # Get the number of correctly classified samples
                    corr_cl = sum(pred == y_gt).cpu().numpy()
                    acc_untargeted[tm_idx, t // args.attack_log_interval] += corr_cl
                    # Get the number of correctly targeted samples
                    corr_tar= sum(pred == y_tar).cpu().numpy()
                    acc_targeted[tm_idx, t // args.attack_log_interval] += corr_tar
                    # Calc l2 norm of delta
                    l2 = torch.sum(torch.norm(delta.reshape(delta.size(0), -1), p=2, dim=1)).detach().cpu().numpy()
                    norm_l2[tm_idx, t // args.attack_log_interval] += l2
                    # Calc linf norm of delta
                    linf = torch.sum(torch.norm(delta.reshape(delta.size(0), -1), p=np.inf, dim=1)).detach().cpu().numpy()
                    norm_linf[tm_idx, t // args.attack_log_interval] += linf


        num_samples += len(x)
        _logger.info('\n-- Untargeted ASR --')
        for tm_idx, tm in enumerate(args.target_model):
            _logger.info('{} -> {}'.format(args.source_model, tm))
            _logger.info('{}'.format(1.-acc_untargeted[tm_idx]/num_samples))

        _logger.info('\n-- Targeted Accuracy --')
        for tm_idx, tm in enumerate(args.target_model):
            _logger.info('{} -> {}'.format(args.source_model, tm))
            _logger.info('{}'.format(acc_targeted[tm_idx]/num_samples))

        _logger.info('\n-- L2 norm --')
        for tm_idx, tm in enumerate(args.target_model):
            _logger.info('{} -> {}'.format(args.source_model, tm))
            _logger.info('{}'.format(norm_l2[tm_idx]/num_samples))

        _logger.info('\n-- Linf norm --')
        for tm_idx, tm in enumerate(args.target_model):
            _logger.info('{} -> {}'.format(args.source_model, tm))
            _logger.info('{}'.format(norm_linf[tm_idx]/num_samples))

        if batch_idx == last_idx:
            break

    # Results
    _logger.info('\n-- Untargeted ASR --')
    for tm_idx, tm in enumerate(args.target_model):
        _logger.info('{} -> {}'.format(args.source_model, tm))
        _logger.info('{}'.format(1.-acc_untargeted[tm_idx]/num_samples))

    _logger.info('\n-- Targeted Accuracy --')
    for tm_idx, tm in enumerate(args.target_model):
        _logger.info('{} -> {}'.format(args.source_model, tm))
        _logger.info('{}'.format(acc_targeted[tm_idx]/num_samples))
    
    _logger.info('\n-- L2 norm --')
    for tm_idx, tm in enumerate(args.target_model):
        _logger.info('{} -> {}'.format(args.source_model, tm))
        _logger.info('{}'.format(norm_l2[tm_idx]/num_samples))

    _logger.info('\n-- Linf norm --')
    for tm_idx, tm in enumerate(args.target_model):
        _logger.info('{} -> {}'.format(args.source_model, tm))
        _logger.info('{}'.format(norm_linf[tm_idx]/num_samples))

    model_string=''
    untargeted_string=''
    targeted_string=''
    untar_tar_string=''
    l2_norm_string = ''
    linf_norm_string = ''
    for tm_idx, tm in enumerate(args.target_model):     
        model_string += tm + ' '
        untargeted_string += '{} '.format(1. - acc_untargeted[tm_idx, -1] / num_samples)
        targeted_string += '{} '.format(acc_targeted[tm_idx, -1] / num_samples)
        untar_tar_string += '{}/{} '.format(1. - acc_untargeted[tm_idx, -1] / num_samples, acc_targeted[tm_idx, -1] / num_samples)
        l2_norm_string += '{} '.format(norm_l2[tm_idx, -1] / num_samples)
        linf_norm_string += '{} '.format(norm_linf[tm_idx, -1] / num_samples)

    _logger.info('{}'.format(model_string))
    _logger.info('-- Untargeted ASR --')
    _logger.info(untargeted_string)
    _logger.info('-- Targeted Accuracy --')
    _logger.info(targeted_string)
    _logger.info('-- Untargeted ASR / Targeted Accuracy --')
    _logger.info(untar_tar_string)
    _logger.info('-- L2 norm --')
    _logger.info(l2_norm_string)
    _logger.info('-- Linf norm --')
    _logger.info(linf_norm_string)
    
if __name__ == '__main__':
    main()
