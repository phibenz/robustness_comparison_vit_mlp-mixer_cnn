import os, csv, logging
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torchvision import models as torchmodels
from models.modeling import VisionTransformer, CONFIGS
import timm

def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel'])-1 )
            label_tar_list.append( int(row['TargetClass'])-1 )

    return image_id_list,label_ori_list,label_tar_list

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
    
    def forward(self, x):
        lo = self.model.forward(x)
        if isinstance(lo, (tuple, list)):
           lo = lo[0]
        return lo

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()

        self.mean = torch.Tensor(mean).reshape(1,3,1,1)
        self.std = torch.Tensor(std).reshape(1,3,1,1)

    def forward(self, x):
        return (x - self.mean.type_as(x)) / self.std.type_as(x)

class Unnormalize(nn.Module):
    def __init__(self, mean, std):
        super(Unnormalize, self).__init__()

        self.mean = torch.Tensor(mean).reshape(1,3,1,1)
        self.std = torch.Tensor(std).reshape(1,3,1,1)

    def forward(self, x):
        return (x * self.std.type_as(x)) + self.mean.type_as(x)

def get_logger(path, filename='log.txt'):
    logger = logging.getLogger('logbuch')
    logger.setLevel(level=logging.DEBUG)
    
    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh_formatter = logging.Formatter('%(message)s')
    sh.setFormatter(sh_formatter)
    
    # File handler
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setLevel(level=logging.DEBUG)
    fh_formatter = logging.Formatter('%(message)s')
    fh.setFormatter(fh_formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def get_timestamp():
    ISOTIMEFORMAT='%Y%m%d_%H%M%S_%f'
    timestamp = '{}'.format(datetime.utcnow().strftime( ISOTIMEFORMAT)[:-3])
    return timestamp

def one_hot(class_labels, num_classes):
    class_labels = class_labels.cpu()
    return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.).cuda()

def get_model(model_name):
    if model_name.startswith('ViT'):
        if model_name.endswith('-224'):
            config = CONFIGS[model_name.rstrip('-224')]
            model = VisionTransformer(config, 224, zero_head=False, num_classes=1000).eval()
        else:
            config = CONFIGS[model_name]
            model = VisionTransformer(config, 384, zero_head=False, num_classes=1000).eval()
        model.load_from(np.load('./checkpoints/{}.npz'.format(model_name)))
    elif model_name=='inception_v3':
        model = torchmodels.__dict__[model_name](pretrained=True, transform_input=True).eval()
    elif model_name in ['swsl_resnet18', 'swsl_resnet50', 'mixer_b16_224', 'mixer_l16_224']:
        model = timm.create_model(model_name, pretrained=True).eval()
    else:
        model = torchmodels.__dict__[model_name](pretrained=True).eval()
    return model