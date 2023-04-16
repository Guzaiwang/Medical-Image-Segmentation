import builtins
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable as V
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
# from torch.utils.tensorboard import SummaryWriter

from training.utils import update_ema_variables
from training.losses import DiceLoss, Dice_bce_loss
from training.validation import validation
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result, 
    get_optimizer, 
    filter_validation_results
)
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings

import matplotlib.pyplot as plt

from utils import (
    configure_logger,
    save_configure,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
)

import types
import collections
from random import shuffle

warnings.filterwarnings("ignore", category=UserWarning)


def train_net(net, args, ema_net=None):

    ################################################################################
    # Dataset Creation
    trainset = get_dataset(args, mode='train')
    
    trainLoader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        pin_memory=True, 
        num_workers=args.num_workers
    )

    testset = get_dataset(args, mode='test')
    testLoader = data.DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)
    
    logging.info(f"Created Dataset and DataLoader")

    ################################################################################
    optimizer = get_optimizer(args, net)

    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)

    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda().float())
    criterion = Dice_bce_loss()
    criterion_dl = DiceLoss()
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ################################################################################
    # Start training

    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")
        
        epoch_loss_value = train_epoch(trainLoader, net, ema_net, optimizer, epoch, criterion, criterion_dl, scaler, args)
        print(epoch_loss_value)
        ########################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = ema_net if args.ema else net 
        
        # save the latest checkpoint, including net, ema_net, and optimizer
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
            'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_latest.pth")



def train_epoch(trainLoader, net, ema_net, optimizer, epoch, criterion, criterion_dl, scaler, args):
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    progress = ProgressMeter(
        len(trainLoader) if args.dimension=='2d' else args.iter_per_epoch, 
        [batch_time, epoch_loss], 
        prefix="Epoch: [{}]".format(epoch+1),
    )   
    
    net.train()

    tic = time.time()
    iter_num_per_epoch = 0 
    for i, inputs in enumerate(trainLoader):
        img, label = inputs[0], inputs[1].float()
        img = V(img.cuda())
        label = V(label.cuda())

        # uncomment this for visualize the input images and labels for debug
        '''
        img = img.cpu()
        print(img.mean())
        label = label.cpu()
        for idx in range(img.shape[0]):
            plt.subplot(3,2,1)
            plt.imshow(img[idx, 0, 64, :, :].numpy())
            plt.subplot(3,2,2)
            plt.imshow(label[idx, 0, 64, :, :].numpy())
            
            plt.subplot(3,2,3)
            plt.imshow(img[idx, 0, :, 64, :].cpu().numpy())
            plt.subplot(3,2,4)
            plt.imshow(label[idx, 0, :, 64, :].numpy())
            
            plt.subplot(3,2,5)
            plt.imshow(img[idx, 0, :, :, 64].cpu().numpy())
            plt.subplot(3,2,6)
            plt.imshow(label[idx, 0, :, :, 64].numpy())
           

            plt.savefig('./result/PtranslateX_idx%d.png'%idx)

            #plt.show()
        '''
        step = i + epoch * len(trainLoader) # global steps
    
        optimizer.zero_grad()
        
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = net(img)

                loss = 0

                if isinstance(result, tuple) or isinstance(result, list):
                    # if use deep supervision, add all loss together
                    for j in range(len(result)):
                        loss += args.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
                else:
                    loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            result = net(img)
            loss = 0 
            if isinstance(result, tuple) or isinstance(result, list):
                # If use deep supervision, add all loss together 
                for j in range(len(result)):
                    loss += args.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
            else:
                # loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)
                loss = criterion(label, result)


            loss.backward()
            optimizer.step()
        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        epoch_loss.update(loss.item(), img.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)

        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break 

    return epoch_loss
def get_parser():
    parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
    parser.add_argument('--dataset', type=str, default='acdc', help='dataset name')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile, only supported by pytorch2.0')

    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args
    


def init_network(args):
    net = get_model(args, pretrain=args.pretrain)

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None
    
    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)
    
    if args.torch_compile:
        net = torch.compile(net)
    return net, ema_net 


if __name__ == '__main__':
    
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.log_path = args.log_path + '%s/'%args.dataset
    

    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)
        torch.backends.cudnn.benchmark = True

        
    args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
    os.makedirs(args.cp_dir, exist_ok=True)
    configure_logger(0, args.cp_dir+f"/exp_results.txt")
    save_configure(args)
    logging.info(
        f"\nDataset: {args.dataset},\n"
        + f"Model: {args.model},\n"
        + f"Dimension: {args.dimension}"
    )

    net, ema_net = init_network(args)

    net.cuda()
    if args.ema:
        ema_net.cuda()
    logging.info(f"Created Model")
    train_net(net, args, ema_net)

    sys.exit(0)
