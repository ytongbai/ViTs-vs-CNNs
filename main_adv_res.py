# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from functools import partial
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import scaler

from datasets import build_dataset
from engine_accum import *
from models.losses import DistillationLoss
from samplers import RASampler
# import models
import utils

# Add from AdvProp 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import math
import os
import shutil
from tensorboardX import SummaryWriter


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import _LRScheduler

from models.attacker import NoOpAttacker, PGDAttacker

from models.ghost_bn import GhostBN2D_ADV

from models.affine import Affine
import global_val

def get_args_parser():
    parser = argparse.ArgumentParser('AdvTrans training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--activation', default='relu', type = str)
    parser.add_argument('--forward-time', default = 8, type = int)
    parser.add_argument('--normalize', default='adv', type = str, choices=['adv', 'clean'])

    parser.add_argument('--nb_classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--sing', default='none', type = str, choices=['none', 'singbn', 'singbn', 'singgbn'])


    parser.add_argument('--adjust_lr', default = 512, type = int)

    #  attacker options
    parser.add_argument('--attack-iter', help='Adversarial attack iteration', type=int, default=0)
    parser.add_argument('--attack-epsilon', help='Adversarial attack maximal perturbation', type=float, default=1.0)
    parser.add_argument('--attack-step-size', help='Adversarial attack step size', type=float, default=1.0)
    
    
    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') #
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--accumulative', action='store_true')
    parser.add_argument('--no-accumulative', action='store_false', dest='accumulative')
    parser.set_defaults(accumulative=False)
    

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    parser.add_argument('--prob_start_from_clean', default=0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


    

def main(args):
    global_val.gbn_forward_time = args.forward_time
#     print(args.forward_time)
    global_val.ratio_=int(args.batch_size / 64)
#                     from models.advresnet_gbn import Affine
    from models.advresnet_gbn_gelu import Affine
    if args.model.startswith('resnet'):
        if args.sing == 'singbn':
            import models.advresnet as advres
        elif args.sing == 'singgbn':
            if args.activation == 'relu':
                import models.advresnet_gbn as advres
            elif args.activation == 'gelu':
                import models.advresnet_gbn_gelu as advres
    else:
        import models.models
    
    if args.attack_iter == 0:
        train_attacker = NoOpAttacker()
    else:
        train_attacker = PGDAttacker(args.attack_iter, args.attack_epsilon, args.attack_step_size, prob_start_from_clean=args.prob_start_from_clean)
    eval_attacker_0 = NoOpAttacker()
    eval_attacker_5 = PGDAttacker(5, 4, 1, prob_start_from_clean=0.0)
    eval_attacker_10 = PGDAttacker(10, 4, 1, prob_start_from_clean=0.0)
    eval_attacker_50 = PGDAttacker(50, 4, 1, prob_start_from_clean=0.0)
    eval_attacker_100 = PGDAttacker(100, 4, 1, prob_start_from_clean=0.0)

    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank() + 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(0.25 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=1000)
    
    
    
    
    
    if args.model.startswith('resnet'):
        if args.sing == 'singbn':
            norm_layer = nn.BatchNorm2d
        elif args.sing == 'singgbn':
            if args.forward_time == 8:
                norm_layer = EightBN 
            elif args.forward_time == 4:
                norm_layer = FourBN 

    else:
        norm_layer = None

        if args.sing == 'singgn2':
            norm_layer = SingGN_2
        elif args.sing == 'singln':
            norm_layer = SingLN
    
    print(f"Creating model: {args.model}")
    if args.model.startswith('resnet'):
        model = advres.__dict__[args.model](norm_layer=norm_layer)
        
    else:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None, #
            norm_layer = norm_layer,
    #         norm = args.norm,
        )
    
    model.set_attacker(train_attacker)
    model.set_mix(False)
        
    if args.sing != 'none':
        model.set_sing(True)
    else:
        model.set_sing(False)
    
    if mixup_active:
        model.set_mixup_fn(True)
    else:
        model.set_mixup_fn(False)
    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() * 4 / args.adjust_lr
    args.lr = linear_scaled_lr
    
    optimizer = create_optimizer(args, model_without_ddp)
    
    loss_scaler = scaler.AccumulativeScaler(forward_time=args.forward_time)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        # clean optimizer, only load weight.
#         if False:
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        model.module.set_attacker(eval_attacker_0)
        test_stats = evaluate(data_loader_val, model, device)
        print('Final Test_attacker_step_0', test_stats["acc1"])

        model.module.set_attacker(eval_attacker_5)
        test_stats = evaluate(data_loader_val, model, device)
        print('Final Test_attacker_step_5', test_stats["acc1"])
        
        model.module.set_attacker(eval_attacker_50)
        test_stats = evaluate(data_loader_val, model, device)
        print('Final Test_attacker_step_50', test_stats["acc1"])

        model.module.set_attacker(eval_attacker_100)
        test_stats = evaluate(data_loader_val, model, device)
        print('Final Test_attacker_step_100', test_stats["acc1"])
        
        return
    
    
    # Train and val
    writer = SummaryWriter(log_dir=args.output_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_0 = 0.0
    max_accuracy_5 = 0.0
    test_stats_0 = {}
    test_stats_5 = {}
    test_stats = {}
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        model.module.set_attacker(train_attacker)
        if args.forward_time == 4:
            train_stats = train_one_epoch_4(
                        model, criterion, data_loader_train,
                        optimizer, device, epoch, loss_scaler,
                        args.clip_grad, model_ema, mixup_fn,
                        set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
                        mix=False, num_classes=args.nb_classes,)
        elif args.forward_time == 8:
            train_stats = train_one_epoch_8(
                        model, criterion, data_loader_train,
                        optimizer, device, epoch, loss_scaler,
                        args.clip_grad, model_ema, mixup_fn,
                        set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
                        mix=False, num_classes=args.nb_classes,)
#         print(train_stats)
        writer.add_scalar('Train/loss', train_stats['loss'], epoch)
        writer.add_scalar('Train/lr', train_stats["lr"], epoch)
        
        
        if epoch >= 0:
            lr_scheduler.step(epoch)
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_{:04d}.pth'.format(epoch)]
            if args.model_ema == True:
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            else:
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
#                         'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
        
            if True: #epoch<15 or (epoch % 5 == 0 and args.attack_iter!=0):
                model.module.set_attacker(eval_attacker_0)
                test_stats_0 = evaluate(data_loader_val, model, device)
                writer.add_scalar('Test/loss_0', test_stats_0['loss'], epoch)
                writer.add_scalar('Test/acc1_0', test_stats_0['acc1'], epoch)
                print(f"Accuracy of the eval_attacker_0 on the {len(dataset_val)} test images: {test_stats_0['acc1']:.1f}%")
                max_accuracy_0 = max(max_accuracy_0, test_stats_0["acc1"])
                print(f'Max accuracy_0: {max_accuracy_0:.2f}%')
            if epoch % 5 == 0 and args.attack_iter!=0: 
                model.module.set_attacker(eval_attacker_5)
                test_stats_5 = evaluate(data_loader_val, model, device)
                writer.add_scalar('Test/loss_5', test_stats_5['loss'], epoch)
                writer.add_scalar('Test/acc1_5', test_stats_5['acc1'], epoch)
                print(f"Accuracy of the eval_attacker_5 on the {len(dataset_val)} test images: {test_stats_5['acc1']:.1f}%")
                max_accuracy_5 = max(max_accuracy_5, test_stats_5["acc1"])
                print(f'Max accuracy_5: {max_accuracy_5:.2f}%')
        else:
            test_stats = evaluate(data_loader_val, model, device)
        
            writer.add_scalar('Test/loss', test_stats['loss'], epoch)
            writer.add_scalar('Test/acc1', test_stats['acc1'], epoch)

            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')
              
              
              
        
        if args.sing == 'none':
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_0_{k}': v for k, v in test_stats_0.items()},
                         **{f'test_5_{k}': v for k, v in test_stats_5.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()
    
class FourBN(nn.Module):
    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2,  affine=False, sync_stats=False, **kwargs):
        super(FourBN, self).__init__()
        virtual2actual_batch_size_ratio = global_val.ratio_
        
        self.bn0 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn1 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn2 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn3 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        
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
        virtual2actual_batch_size_ratio = global_val.ratio_
        
        self.bn0 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn1 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn2 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn3 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn4 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn5 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn6 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn7 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
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

class SingLN(nn.LayerNorm):
    def __init__(self, normalized_shape, eps = 1e-6, elementwise_affine = True):
        super(SingLN, self).__init__(normalized_shape, eps, elementwise_affine)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = super(SingLN, self).forward(input)
        elif self.batch_type == 'clean':
            input = super(SingLN, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            input = super(SingLN, self).forward(input)
        return input

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Advtraining and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)