#!/usr/bin/env python
# coding: utf-8

import torch
import os
from torchvision import transforms
from PIL import Image

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, args, is_train=True):
        self.args = args
        self.is_train = is_train
        self.resize = args.resize

        phase = 'train' if self.is_train else 'test'
        self.x, self.y = [], []

        data_path = f'data'
        phase_dir = os.path.join(data_path, args.data_type, phase)
        gt_dir = os.path.join(data_path, args.data_type, 'ground_truth')

        img_list = []
        for data_type in os.listdir(phase_dir):
            data_type_dir = os.path.join(phase_dir, data_type)
            imgs = []
            imgs = sorted(os.listdir(data_type_dir))
            for img in imgs:
                img_dir = os.path.join(data_type_dir, img)
                self.x.append(img_dir)
                if is_train:
                    gt_name = os.path.splitext(img)[0] + '.png'
                    gt_path = os.path.join(gt_dir, 'train', gt_name)
                    self.y.append(gt_path)
                else:
                    if data_type == 'good':
                        self.y.append(0)
                    else:
                        ano_name = os.path.splitext(img)[0] + '_mask.png'
                        ano_path = os.path.join(gt_dir, data_type, ano_name)
                        self.y.append(ano_path)

        self.transform_x = transforms.Compose([
            transforms.Resize((self.resize, self.resize), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.transform_y = transforms.Compose([
            transforms.Resize((self.resize, self.resize), Image.ANTIALIAS),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB') # 受け取ったアドレスを開く
        x = self.transform_x(x)
        if y == 0:
            y = torch.zeros([1, self.resize, self.resize])
        else:
            y = Image.open(y).convert('1')
            y = self.transform_y(y)

        return x, y

    def __len__(self):
        return len(self.x)