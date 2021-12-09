# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from models.losses import DistillationLoss
import utils
import time


def train_one_epoch_4(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, mix=False, num_classes=1000):
   
    model.train(set_training_mode)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for ii, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        if ii % 4 == 0:
            model.module.set_bn_num(0)
        elif ii % 4 == 1:
            model.module.set_bn_num(1)
        elif ii % 4 == 2:
            model.module.set_bn_num(2)
        elif ii % 4 == 3:
            model.module.set_bn_num(3)

        with torch.cuda.amp.autocast():
            outputs, targets = model(samples, targets)
            loss = criterion(samples, outputs, targets)
        
        loss_value = loss.item()
        
        loss = loss / 4        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)            
            
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler.set_iters(ii)
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        
        if ((ii+1)% 4)==0:
            
            optimizer.zero_grad()


        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
            
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
    
def train_one_epoch_8(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, mix=False, num_classes=1000):
   
    model.train(set_training_mode)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for ii, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        if ii % 8 == 0:
            model.module.set_bn_num(0)
        elif ii % 8 == 1:
            model.module.set_bn_num(1)
        elif ii % 8 == 2:
            model.module.set_bn_num(2)
        elif ii % 8 == 3:
            model.module.set_bn_num(3)
        elif ii % 8 == 4:
            model.module.set_bn_num(4)
        elif ii % 8 == 5:
            model.module.set_bn_num(5)
        elif ii % 8 == 6:
            model.module.set_bn_num(6)
        elif ii % 8 == 7:
            model.module.set_bn_num(7)

        with torch.cuda.amp.autocast():
            outputs, targets = model(samples, targets)
            loss = criterion(samples, outputs, targets)
        
        loss_value = loss.item()
        
        loss = loss / 8        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)                   

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler.set_iters(ii)
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        
        if ((ii+1)% 8)==0:
            
            optimizer.zero_grad()


        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
            
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    

# @torch.no_grad()
def evaluate(data_loader, model, device):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, target = model(images, target)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#         break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
