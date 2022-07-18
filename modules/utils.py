import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
import csv

def LossToFig(loss_list:dict, save_path:str, y_lim=0.4):
    fig = plt.figure(figsize=(6,6), facecolor='w')
    for label, loss in loss_list.items():
        plt.plot(loss, label=label)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0,y_lim)
    plt.grid()
    fig.savefig(save_path)
    plt.clf()
    plt.close()

def LossToCsv(loss_list, save_path:str):
    with open(save_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(loss_list)

def CsvToLoss(path:str):
    with open(path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for rows in reader:
            loss_list = [float(s) for s in rows]
        
    return loss_list

def CalcIoU(gt_mask, pred_mask):
    gt_mask = gt_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    pr = gt_mask * pred_mask # 要素積
    add = gt_mask + pred_mask
    add[add > 1] = 1

    iou_list  = []
    for p, a in zip(pr, add):
        iou = p.sum()/a.sum()
        iou_list.append(iou)

    return iou_list

def CalcPixPr(args, logger, gts, scores):
    precision, recall, thresholds = metrics.precision_recall_curve(gts.flatten(), scores.flatten())
    args.pixel_prauc = metrics.auc(recall, precision)
    logger.info(f'pixel PRAUC: {args.pixel_prauc:.4f}')
    plt.plot(recall, precision, label=f'{args.data_type} pixel_PRAUC: {args.pixel_prauc:.3f}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.ad_dir, args.data_type + '_pr_curve.png'), dpi=100)
    plt.clf()
    plt.close()

    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    return threshold

def CalcPixRoc(args, logger, gts, scores):
    fpr, tpr, _ = metrics.roc_curve(gts.flatten(), scores.flatten())
    args.pixel_rocauc = metrics.roc_auc_score(gts.flatten(), scores.flatten())
    logger.info(f'pixel AUROC: {args.pixel_rocauc:.4f}')
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')
    plt.plot(fpr, tpr, label=f'{args.data_type} AURUC: {args.pixel_rocauc:.4f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.ad_dir, args.data_type + '_roc_curve.png'), dpi=100)
    plt.clf()
    plt.close()

def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x

def PlotFig(args, test_img, masked_img, recon_imgs, scores, gts, threshold):
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    loop = tqdm(zip(test_img, masked_img, recon_imgs, gts, scores), total=len(test_img), desc='Plt', unit='plt')
    for i, (img, masked, recon_img, gt, mask) in enumerate(loop):
        img = denormalization(img)
        masked = denormalization(masked)
        recon_img = denormalization(recon_img)
        gt = gt.transpose(1, 2, 0).squeeze()
        heat_map = mask * 255
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        #kernel = morphology.disk(4)
        #mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 8, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        #norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(masked)
        ax_img[1].title.set_text('Masked')
        ax_img[2].imshow(recon_img)
        ax_img[2].title.set_text('Reconst')
        ax_img[3].imshow(gt, cmap='gray')
        ax_img[3].title.set_text('GroundTruth')
        #ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[4].imshow(img, cmap='gray', interpolation='none')
        ax_img[4].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', vmin=vmin, vmax=vmax)
        ax_img[4].title.set_text('HeatMap Lay')
        ax_img[5].imshow(heat_map, cmap='jet', interpolation='none', vmin=vmin, vmax=vmax)
        ax_img[5].title.set_text('HeatMap')
        ax_img[6].imshow(mask, cmap='gray')
        ax_img[6].title.set_text('Predicted mask')
        ax_img[7].imshow(vis_img)
        ax_img[7].title.set_text('Segmentation result')

        fig_img.savefig(os.path.join(args.ad_dir, f'{args.data_type}_{i}.png'), dpi=100)
        plt.close()