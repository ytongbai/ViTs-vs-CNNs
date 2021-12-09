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
# import torchvision.models as models
from torch.optim.lr_scheduler import _LRScheduler

from models.ghost_bn import GhostBN2D
import models.resnet_gbn


from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from util.imagenet_a import indices_in_1k
from tensorboardX import SummaryWriter

import models
from pim.timm.models import create_model

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
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--warm', default=5, type=int,
                    help='# of warm up epochs')
parser.add_argument('--warm_lr', default=0., type=float,
                    help='warm up start learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/tmp/checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load', default='', type=str,
                    help='load the checkpoint for finetune / evaluation')
parser.add_argument('--finetune', action='store_true',
                    help='ignore aux bn when finetune')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',)
parser.add_argument('--ckpt', default = '')
#                     choices=model_names,
#                     help='model architecture: ' +
#                          ' | '.join(model_names) +
#                          ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29,
                    help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32,
                    help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4,
                    help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4,
                    help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Core of debiased training
parser.add_argument('--style', action='store_true',
                    help='use style transfer')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='alpha value for style transfer')
parser.add_argument('--label-gamma', default=0.8, type=float,
                    help='gamma in Eq. (1) in paper')
parser.add_argument('--mixbn', action='store_true',
                    help='whether using auxiliary batch normalization')
parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cos'])
parser.add_argument('--multi_grid', action='store_true',
                    help='use downsampled images as input of style transfer for speed up training process')
parser.add_argument('--min_size', default=112, type=int,
                    help='the min size of down sampled images')

# Combine with other data augmentations
parser.add_argument('--mixup', default=0., type=float,
                    help='mixup hyper-parameter')
parser.add_argument('--cutmix', default=0., type=float,
                    help='cutmix hyper-parameter')

# Evaluation options
parser.add_argument('--evaluate_imagenet_c', action='store_true',
                    help="for evaluate Imagenet-C")
parser.add_argument('--already224', action='store_true',
                    help="skip crop and resize if inputs are already 224x224 (for evaluate Stylized-ImageNet)")
parser.add_argument('--imagenet-a', action='store_true',
                    help="mapping the 1k labels to 200 labels (for evaluate ImageNet-A)")
parser.add_argument('--FGSM', action='store_true',
                    help="evalute FGSM robustness")
parser.add_argument('--mocov2', action = 'store_true')


# Parametes for deit

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--sing', default='singbn', type=str)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

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

best_acc = 0  # best test accuracy


def main():
    
    if args.sing == 'singbn':
        norm_layer = nn.BatchNorm2d
    elif args.sing == 'singgbn':
        norm_layer = GhostBN2D
    global best_acc, state
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transforms = [
        transforms.ToTensor(),
        normalize,
    ]
    if not args.already224:
        # This option is for evaluating Stylized-ImageNet, which is already 224x224
        val_transforms = [transforms.Scale(256), transforms.CenterCrop(224)] + val_transforms
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(val_transforms)),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True) if not args.evaluate_imagenet_c else None

    # create model
    if args.arch.startswith('resnext'):
        norm_layer = MixBatchNorm2d if args.mixbn else None
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
            num_classes=args.num_classes,
            norm_layer=norm_layer
        )
    elif args.arch.startswith('vit'):
        model = create_model(
        args.arch,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None, #
#         norm = 'layer', #############change here. Also, check the codebase, make sure it works on the original version too.
        )
        a = torch.load(args.ckpt)
        model.load_state_dict(a['model'])
    elif args.arch.startswith('regnet'):
        model = timm.create_model(args.arch, pretrained=False)
        a = torch.load(args.ckpt)['model']
        model.load_state_dict(a)
    elif args.arch.startswith('deit'):
        model = create_model(
        args.arch,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None, #
        norm = 'layer', #############change here. Also, check the codebase, make sure it works on the original version too.
    )
        a = torch.load(args.ckpt)
        model.load_state_dict(a['model'])
    elif 'resnet' in args.arch: #.startswith('resnet'):
        print("=> creating model '{}'".format(args.arch))
        model = resnet_gbn.__dict__[args.arch](norm_layer=norm_layer)
        
        a = torch.load(args.ckpt)
        model.load_state_dict(a['model'])

            
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda, args.FGSM)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    if args.evaluate_imagenet_c:
        print("Evaluate ImageNet C")
        distortions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]

        error_rates = []
        for distortion_name in distortions:
            rate = show_performance(distortion_name, model, criterion, start_epoch, use_cuda)
            error_rates.append(rate)
            print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))
        print(distortions)
        print(error_rates)
        print(np.mean(error_rates))
        return

def test(val_loader, model, criterion, epoch, use_cuda, FGSM=False):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            if args.imagenet_a:
                outputs = outputs[:, indices_in_1k]
            loss = criterion(outputs, targets).mean()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def show_performance(distortion_name, model, criterion, start_epoch, use_cuda):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    errs = []

    for severity in range(1, 6):
        valdir = os.path.join(args.data, distortion_name, str(severity))
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                # transforms.Scale(256),
                # transforms.CenterCrop(224), # already 224 x 224
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)

        errs.append(1. - test_acc / 100.)

    print('\n=Average', tuple(errs))
    return np.mean(errs)


if __name__ == '__main__':
    main()
