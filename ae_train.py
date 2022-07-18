#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
from torch import optim
from torchvision import utils
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
from modules.loader import DatasetLoader
from modules.loader import DatasetLoader
from modules.initializer import init_weight
from modules.losses import SSIM
from modules.utils import LossToFig, LossToCsv, CalcIoU

def train(args, logger, model):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    dataset = DatasetLoader(args, is_train=True)

    kf = KFold(n_splits=args.split_len, shuffle=True, random_state=args.seed)
    for k, (train_index, val_index) in enumerate(kf.split(dataset)):
        if k != args.split_num:
            continue
        result_dir = os.path.join(args.result_dir, f'fold{k}')
        pic_dir = os.path.join(result_dir, 'pic')
        os.makedirs(pic_dir, exist_ok=True)

        train_dataset = Subset(dataset, train_index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
        logger.info(f'Train [{k}] : {len(train_dataset)}({len(train_loader)})')

        val_dataset = Subset(dataset, val_index)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=False, drop_last=False)
        logger.info(f'Val [{k}] : {len(val_dataset)}({len(val_loader)})')

        for vis_inputs, _ in val_loader:
            vis_inputs = vis_inputs[:25].to(args.device)
            utils.save_image(vis_inputs, pic_dir + f'/input.png', nrow=5, normalize=True)
            break

        init_weight(model, init_type=args.init_type)
        
        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr}
        ])

        if args.start_epoch != 1:
            pt_model = torch.load(f'{args.model_path}_{args.start_epoch}.pt')
            model.load_state_dict(pt_model['model'])
            optimizer.load_state_dict(pt_model['opt'])

        if args.scheduler == 'LR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)
            logger.info('Scheduler : MultiStepLR')

        loss_l2 = torch.nn.modules.MSELoss()
        loss_ssim = SSIM()

        l2_loss_list = []
        ssim_loss_list = []
        val_loss_list = []
        for epoch in tqdm(range(args.start_epoch, args.epochs + 1), unit='epoch', leave=False):
            model.train()
            loop = tqdm(train_loader, unit='batch', desc=f'Train [Epoch {epoch:>3}]')

            l2_losses = []
            ssim_losses = []
            for _, (inputs, _) in enumerate(loop):
                inputs = inputs.to(args.device)

                outputs = model(inputs)

                l2_loss = loss_l2(outputs, inputs)
                ssim_loss = loss_ssim(outputs, inputs)

                loss = l2_loss*(1-args.ssim_loss_rate) + ssim_loss*(args.ssim_loss_rate)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                l2_losses.append(l2_loss.item())
                ssim_losses.append(ssim_loss.item())
                
            if args.scheduler == 'MultiStepLR':
                scheduler.step()
                
            l2_loss_list.append(np.average(l2_losses))
            ssim_loss_list.append(np.average(ssim_losses))
            logger.info(f'Train [Epoch {epoch}/{args.epochs}] L2_loss : {np.average(l2_losses)}, SSIM_loss : {np.average(ssim_losses)}')

            with torch.no_grad():
                model.eval()
                val_loop = tqdm(val_loader, unit='batch', desc=f'Val [Epoch {epoch:>3}]')

                val_losses = []
                for i, (inputs, _) in enumerate(val_loop):
                    inputs = inputs.to(args.device)

                    outputs = model(inputs)

                    l2_loss = loss_l2(outputs, inputs)
                    ssim_loss = loss_ssim(outputs, inputs)

                    loss = l2_loss*(1-args.ssim_loss_rate) + ssim_loss*(args.ssim_loss_rate)

                    val_losses.append(loss.item())

                    if i == 0:
                        utils.save_image(outputs[:25], pic_dir + f'/out_{epoch}.png', nrow=5, normalize=True)

                val_loss_list.append(np.average(val_losses))
                logger.info(f'Val {k} [Epoch {epoch}/{args.epochs}] Loss : {val_loss_list[-1]}')

            if epoch % 50 == 0 and epoch != args.start_epoch:
                torch.save({'model':model.state_dict(),'opt':optimizer.state_dict()}, f'{args.model_path}/fold{k}_{epoch}.pt')
                logger.info(f'AE Model[{k}] epoch({epoch}) exported.')

        if epoch % 50 != 0:
            torch.save({'model':model.state_dict(),'opt':optimizer.state_dict()}, f'{args.model_path}/fold{k}_{epoch}.pt')
            logger.info(f'AE Model[{k}] epoch({epoch}) exported.')

        LossToCsv(l2_loss_list, os.path.join(result_dir, 'l2_loss.csv'))
        LossToCsv(ssim_loss_list, os.path.join(result_dir, 'ssim_loss.csv'))
        logger.info(f'Loss[{k}] csv exported.')

        LossToFig({'l2_loss':l2_loss_list, 'ssim_loss':ssim_loss_list}, os.path.join(result_dir, 'train_loss.png'))
        train_loss_list = np.array(l2_loss_list) + np.array(ssim_loss_list)
        LossToFig({'train_loss':train_loss_list, 'val_loss':val_loss_list}, os.path.join(result_dir, 'loss.png'))
        logger.info(f'[{k}] graph exported.')
