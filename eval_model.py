import argparse, time, os, sys
import shutil
from collections import OrderedDict
from contextlib import suppress
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import Dataset
from timm.utils import AverageMeter, accuracy

from utils import Normalize, Unnormalize, get_logger, get_timestamp, load_ground_truth, get_model
from config import IMAGENET_PATH, NEURIPS_DATA_PATH, NEURIPS_CSV_PATH

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Evaluate a model')
# Dataset / Model parameters
parser.add_argument('--source-model', nargs="+", default=['resnet101'], help='Name of model to train')
parser.add_argument('--dataset', default='imagenet', type=str, help='Used Dataset')
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
parser.add_argument('--image-size', type=int, default=224, help='Image size')
# Misc
parser.add_argument('--log-interval', type=int, default=50, help='log interval')
parser.add_argument('--workers', type=int, default=8, help='Dataloading workers')
parser.add_argument('--subfolder', default='', type=str, help='Subfolder')
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

    result_path = os.path.join('./output', 'eval', args.subfolder, get_timestamp() + args.postfix)
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
        mean = args.mean = [0.485, 0.456, 0.406]
        std = args.std = [0.229, 0.224, 0.225]

        if args.image_size == 224:
            transform_eval = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        else:
            transform_eval = transforms.Compose([
                    transforms.Resize(args.image_size, Image.BICUBIC),
                    transforms.CenterCrop(args.image_size),
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
        mean = args.mean = [0.485, 0.456, 0.406]
        std = args.std = [0.229, 0.224, 0.225]

        transform_eval = transforms.Compose([
                transforms.Resize(args.image_size),
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
        model = get_model(sm)
        model.cuda()
        source_model.append(model)

    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    # Evaluation
    for model, model_name in zip(source_model, args.source_model):

        ##### Validation
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        unnorm = Unnormalize(mean=args.mean, std=args.std)
        norm_vit = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        model.eval()

        end = time.time()
        last_idx = len(loader_eval) - 1

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader_eval):
                if args.dataset=='neurips':
                    target=target[0]

                input = input.cuda()
                target = target.cuda()
                
                if model_name.startswith('ViT') or model_name.startswith('mixer_') or model_name.startswith('vit_'):
                    input = norm_vit(unnorm(input))

                with amp_autocast():
                    output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = validate_loss_fn(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if batch_idx % args.log_interval == 0:
                    log_name = 'Test'
                    print(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))
                
        val_metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

        #####
        # val_metrics = validate(sm, sm_name, loader_eval, validate_loss_fn, args)
        _logger.info(f"Top-1 accuracy of {model_name}: {val_metrics['top1']:.2f}%")

if __name__ == '__main__':
    main()
