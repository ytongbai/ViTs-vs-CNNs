'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import numpy as np
from PIL import ImageFile
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import math
import os
import shutil
import time
import random
from functools import partial

# pytorch related
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import models
from timm.models import create_model
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from att.autoattack.autoattack import AutoAttack


from ghost_bn import GhostBN2D

from ghost_bn_old import GhostBN2D_Old

import resnet_gbn

import torch.nn.functional as F

upper_limit, lower_limit = 1, 0

# This mean and std is for adversarial training.
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
mu = torch.tensor(mean).view(3,1,1).cuda()
std = torch.tensor(std).view(3,1,1).cuda()



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Affine(nn.Module):

    def __init__(self, width, *args, k=1, **kwargs):
        super(Affine, self).__init__()
        self.bnconv = nn.Conv2d(width,
                              width,
                              k,
                              padding=(k - 1) // 2,
                              groups=width,
                              bias=True)

    def forward(self, x):
        return self.bnconv(x)

    
    
def normalize(X):
    return (X - mu)/std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)



def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            #normalize(X + delta)
            output, _ = model(X+ delta, y)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        k_, _ = model(X, y)
        all_loss = F.cross_entropy(k_, y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(normalize(X))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

class FourBN(nn.Module):
        
    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2, affine=False, sync_stats=False, **kwargs):
        super(FourBN, self).__init__()
        
        self.bn0 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn1 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn2 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn3 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        
        self.bn_type = 'bn0'
        self.aff = Affine(width=num_features, k=1)
        
    def forward(self, input):
        if self.bn_type == 'bn0':
            input = self.bn0(input)
        elif self.bn_type == 'bn1':
            input = self.bn1(input)
        elif self.bn_type == 'bn2':
            input = self.bn2(input)
        elif self.bn_type == 'bn3':
            input = self.bn3(input)
        
        input = self.aff(input)
        return input

class EightBN(nn.Module):
        
    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2, affine=False, sync_stats=False, **kwargs):
        super(EightBN, self).__init__()
        
        self.bn0 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn1 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn2 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn3 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn4 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn5 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn6 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn7 = GhostBN2D_Old(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        
        self.bn_type = 'bn0'
        self.aff = Affine(width=num_features, k=1)
        
    def forward(self, input):
        if self.bn_type == 'bn0':
            input = self.bn0(input)
        elif self.bn_type == 'bn1':
            input = self.bn1(input)
        elif self.bn_type == 'bn2':
            input = self.bn2(input)
        elif self.bn_type == 'bn3':
            input = self.bn3(input)
        elif self.bn_type == 'bn4':
            input = self.bn4(input)
        elif self.bn_type == 'bn5':
            input = self.bn5(input)
        elif self.bn_type == 'bn6':
            input = self.bn6(input)
        elif self.bn_type == 'bn7':
            input = self.bn7(input)
        
        input = self.aff(input)
        return input    

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

model_names = default_model_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='/path/to/imagenet-a', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize (default: 200)')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',)
parser.add_argument('--ckpt', default = '')

parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Evaluation options

parser.add_argument('--sing', default= 'singbn', type = str)
# Parametes for deit

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--eps', type=float, default=4/255, )
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--pretrain-bs', type = int, default = 256)
parser.add_argument('--forward-time', type = int, default = 4)

parser.add_argument('--activation', default= 'relu', type = str)
args = parser.parse_args()

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def evaluate(data_loader, model):
    
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()
    acc = 0
    for i, (images, target) in enumerate(data_loader):
        
        images = images.cuda()
        target = target.cuda()
        output, target = model(images, target)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc += acc1
        print(acc/(i+1))
        

    return acc/(i+1)


def main():
    print(args)
    if args.arch.startswith('resnet'):
        if args.pretrain_bs == 4096:
            if args.activation == 'relu':
                import resnet_gbn_relu_4096 as res
            elif args.activation == 'gelu':
                import resnet_gbn_gelu_4096 as res
        else: 
            import resnet_gbn as res
    else:
        import models
    
    if args.sing == 'singbn':
        norm_layer = nn.BatchNorm2d
    elif args.sing == 'singgbn':
        if args.pretrain_bs == 4096:
            if args.forward_time == 4:
                norm_layer = FourBN
            elif args.forward_time == 8:
                norm_layer = EightBN
        else:  
            norm_layer = GhostBN2D

    valdir = os.path.join(args.data, 'val')

    val_transforms = [
        transforms.ToTensor(),
    ]

    val_transforms = [transforms.Scale(256), transforms.CenterCrop(224)] + val_transforms
    
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_transforms))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True) 

    if args.arch.startswith('deit'):
        model = create_model(
        args.arch,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None, #
        #norm = 'layer',
        )
        a = torch.load(args.ckpt)
        model.load_state_dict(a['model'])

    elif 'resnet' in args.arch: #.startswith('resnet'):
        print("=> creating model '{}'".format(args.arch))
        model = res.__dict__[args.arch](norm_layer=norm_layer)

        a = torch.load(args.ckpt)
        model.load_state_dict(a['model'])        
    
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    if args.arch == 'deit_small_patch16_224_adv':
        model.module.set_sing(True)
        model.module.set_mixup_fn(False)
        model.module.set_mix(False)   



    l = [x for (x, y) in val_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in val_loader]
    y_test = torch.cat(l, 0)

    
    class normalize_model():
        def __init__(self, model):
            self.model_test = model
        def __call__(self, x):
            return self.model_test(normalize(x))
    
    model = normalize_model(model)
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version='standard')
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.test_batch)
    return


if __name__ == '__main__':
    main()
