#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import torch
from test import test
from modules.logger import get_logger
from modules.model import *

def main(args):
    in_ch = 3
    if args.model == 'concat':
        from concat_train import train
        in_ch=5
    elif args.model == 'softmax':
        from softmax_train import train
    elif args.model == 'separate':
        from separate_train import train
    elif args.model == 'ae':
        from ae_train import train
    else:
        print('Model Error')
        sys.exit()

    logger = get_logger(args.result_dir, f'main{args.split_num}.log')
    state = {k: v for k, v in args._get_kwargs()}
    logger.info(state)

    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    logger.info(f'device : {args.device}')

    if args.unet_model == 'unet':
        model_seg = UNet(dim=args.unet_dim).to(args.device)
    else:
        model_seg = UNet2(dim=args.unet_dim).to(args.device)

    if args.ae_model == 'ae':
        model = CAE(dim=args.ae_dim, in_ch=in_ch).to(args.device)
    else:
        model = CAE2(dim=args.ae_dim,in_ch=in_ch).to(args.device)

    if args.train_skip:
        if args.model == 'ae':
            train(args, logger, model)
        else:
            train(args, logger, model_seg, model)
    test(args, logger, model)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Back Remove Autoencoder')
    parser.add_argument('--model', type=str, default='concat')
    parser.add_argument('--dataset', type=str, default='cloth')
    parser.add_argument('--data_type', type=str, default='pants')
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--split_len', type=int, default=5)
    parser.add_argument('--split_num', type=int, default=2)
    parser.add_argument('--init_type', type=str, default='he_uni')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unet', type=bool, default=True)
    parser.add_argument('--unet_model', type=str, default='unet')
    parser.add_argument('--unet_dim', type=int, default=8)
    parser.add_argument('--ae_model', type=str, default='aeI')
    parser.add_argument('--ae_dim', type=int, default=16)
    parser.add_argument('--ssim_loss_rate', type=float, default=0.5)
    parser.add_argument('--lr', action='store', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, default='LR')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--epochs', action='store', type=int, default=200)
    parser.add_argument('--train_skip', action='store_false')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--ssim_rate', type=float, default=0)
    args = parser.parse_args()

    if args.model ==  'ae':
        args.result_dir = os.path.join(args.result_path, f'{args.dataset}{args.resize}{args.scheduler}_ae', f'{args.ae_model}{args.ae_dim}', f'seed{args.seed}_batch{args.batch_size}{args.init_type}')
        args.model_path = os.path.join(args.model_path, f'{args.dataset}{args.resize}{args.scheduler}_ae', f'{args.ae_model}{args.ae_dim}', f'seed{args.seed}_batch{args.batch_size}{args.init_type}')
    else:
        args.result_dir = os.path.join(args.result_path, f'{args.dataset}{args.resize}{args.scheduler}_{args.model}', f'{args.unet_model}{args.unet_dim}_{args.ae_model}{args.ae_dim}', f'seed{args.seed}_batch{args.batch_size}{args.init_type}')
        args.model_path = os.path.join(args.model_path, f'{args.dataset}{args.resize}{args.scheduler}_{args.model}', f'{args.unet_model}{args.unet_dim}{args.ae_model}{args.ae_dim}', f'seed{args.seed}_batch{args.batch_size}{args.init_type}')
    
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    main(args)