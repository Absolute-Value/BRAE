#!/usr/bin/env python
# coding: utf-8

import os
import csv
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from modules.model import UNet
from modules.loader import DatasetLoader
from modules.utils import CsvToLoss, CalcPixPr, CalcPixRoc, PlotFig, denormalization
from modules.losses import SSIM
def test(args, logger, model):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    test_dataset = DatasetLoader(args, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=False, drop_last=False)

    k = args.split_num
    ssim_map = SSIM(is_map=True)
        
    logger.info(f'fold:{k} epoch:{args.epochs}')
    if args.ssim_rate == 0:
        args.ad_dir = os.path.join(args.result_dir, f'fold{k}', f'ad{args.epochs}')
    else:
        args.ad_dir = os.path.join(args.result_dir, f'fold{k}', f'ssim{args.ssim_rate}_{args.epochs}')
    os.makedirs(args.ad_dir, exist_ok=True)
    add_path = f'_Ep{args.epochs}' if args.model == 'separate' else ''
    
    if args.model != 'ae':
        model_seg = UNet(dim=args.unet_dim).to(args.device)
        pt_model_seg = torch.load(f'{args.model_path}{add_path}/fold{k}_{args.epochs}seg.pt')
        model_seg.load_state_dict(pt_model_seg['model'])
    pt_model = torch.load(f'{args.model_path}{add_path}/fold{k}_{args.epochs}.pt')
    model.load_state_dict(pt_model['model'])
    
    imgs = []
    masked_imgs = []
    gt_masks = []
    pred_masks = []
    reconsts = []
    scores = []
    with torch.no_grad():
        if args.model != 'ae':
            model_seg.eval()
        model.eval()
        loop = tqdm(test_loader, unit='batch', desc='Test')

        for _, (inputs, masks) in enumerate(loop):
            imgs.extend(inputs.cpu().numpy())
            gt_masks.extend(masks.cpu().numpy())
            inputs = inputs.to(args.device)

            if args.model == 'ae':
                outputs = model(inputs)
                inputs = getGrayImage(inputs)
            else:
                out_masks = model_seg(inputs)
                out_masks = torch.softmax(out_masks, dim=1)
                out_mask = torch.argmax(out_masks,dim=1)
                pred_masks.extend(out_mask.cpu().numpy())

                out_mask = out_mask.unsqueeze(1)
                masked_in = inputs * out_mask
                masked_imgs.extend(masked_in.cpu().numpy())
            
                if args.model == 'concat':
                    cat_in = torch.cat((inputs, out_masks),dim=1)
                    outputs = model(cat_in)

                elif args.model == 'softmax':
                    out_masks = out_masks[:,1,:,:].unsqueeze(1)
                    to_masked_in = inputs * out_masks
                    outputs = model(to_masked_in)

                elif args.model == 'separate':
                    outputs = model(masked_in)
                
                masked_in = getGrayImage(masked_in)

            reconsts.extend(outputs.cpu().numpy())
            outputs = getGrayImage(outputs)
            
            if args.model == 'ae':
                score = (1-args.ssim_rate) * torch.abs(outputs-inputs) + args.ssim_rate * ssim_map(outputs, inputs)
            else:
                score = (1-args.ssim_rate) * torch.abs(outputs-masked_in) + args.ssim_rate * ssim_map(outputs, masked_in)

            scores.extend(score.cpu().numpy())

        gt_masks = np.asarray(gt_masks)
        scores = np.asarray(scores)
        scores_min = scores.min()
        scores_max = scores.max()
        scores = (scores - scores_min) / (scores_max - scores_min)
        scores = scores.squeeze(1)

        threshold = CalcPixPr(args, logger, gt_masks, scores)
        CalcPixRoc(args, logger, gt_masks, scores)

        if args.model == 'ae':
            ae_plot_fig(args, imgs, reconsts, scores, gt_masks, threshold)
        else:
            PlotFig(args, imgs, masked_imgs, reconsts, scores, gt_masks, threshold)

        

        if os.path.isfile(os.path.join(args.result_dir, f'fold{k}', 'train_iou.csv')):
            if not hasattr(args, 'train_iou'):
                args.train_iou = np.average(CsvToLoss(os.path.join(args.result_dir, f'fold{k}', 'train_iou.csv')))
                args.val_iou = np.average(CsvToLoss(os.path.join(args.result_dir, f'fold{k}', 'val_iou.csv')))
                
            with open(os.path.join(args.result_path, f'{args.model}.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([args.seed, k, args.init_type, args.batch_size, f'{args.unet_model}{args.unet_dim}', f'{args.ae_model}{args.ae_dim}', args.lr, args.epochs, args.train_iou, args.val_iou, args.ssim_rate, args.pixel_prauc, args.pixel_rocauc])

        else:
            with open(os.path.join(args.result_path, f'{args.model}.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                if args.model == 'ae':
                    writer.writerow([args.seed, k, args.init_type, args.batch_size, f'{args.ae_model}{args.ae_dim}', args.ssim_loss_rate, args.lr, args.epochs, args.ssim_rate, args.pixel_prauc, args.pixel_rocauc])
                else:
                    writer.writerow([args.seed, k, args.init_type, args.batch_size, f'{args.unet_model}{args.unet_dim}', f'{args.ae_model}{args.ae_dim}', args.ssim_loss_rate, args.lr, args.epochs, '', '', args.ssim_rate, args.pixel_prauc, args.pixel_rocauc])

def getGrayImage(rgbImg):
    gray = 0.114*rgbImg[:,0,:,:] + 0.587*rgbImg[:,1,:,:] + 0.299*rgbImg[:,2,:,:]
    gray = torch.unsqueeze(gray,1)
    return gray

def ae_plot_fig(args, test_img, recon_imgs, scores, gts, threshold):
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    loop = tqdm(zip(test_img, recon_imgs, gts, scores), total=len(test_img), desc='Plt', unit='plt')
    for i, (img, recon_img, gt, mask) in enumerate(loop):
        img = denormalization(img)
        recon_img = denormalization(recon_img)
        gt = gt.transpose(1, 2, 0).squeeze()
        heat_map = mask * 255
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 7, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', vmin=vmin, vmax=vmax)
        ax_img[3].title.set_text('HeatMap Lay')
        ax_img[4].imshow(heat_map, cmap='jet', interpolation='none', vmin=vmin, vmax=vmax)
        ax_img[4].title.set_text('HeatMap')
        ax_img[5].imshow(mask, cmap='gray')
        ax_img[5].title.set_text('Predicted mask')
        ax_img[6].imshow(vis_img)
        ax_img[6].title.set_text('Segmentation result')

        fig_img.savefig(os.path.join(args.ad_dir, f'{args.data_type}_{i}.png'), dpi=100)
        plt.close()