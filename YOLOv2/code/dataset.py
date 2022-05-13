import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import os.path as osp

cv2.setNumThreads(0)

class CustomDataset(Dataset):
    def __init__(self, data_dir='./data/train', label_dir='./data/labels', transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

        self.img_names = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.n_samples = len(self.img_names)


    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx):
        img_fname = self.img_names[idx]
        label_fname = img_fname.replace('.png', '.txt')

        # Image & Label path
        img_path = osp.join(self.data_dir, img_fname)
        label_path = osp.join(self.label_dir, label_fname)

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load annotations
        bboxes = []
        with open(label_path, 'r') as f:
            while True:
                box_info = f.readline()
                if not box_info:
                    break

                box_info = box_info.strip().split(' ')
                box_info = [float(e) if float(e) != int(float(e)) else int(e) for e in box_info]
                label, x, y, w, h = box_info
                bboxes.append([x, y, w, h, label])

        # Transform
        if self.transform:
            transformed = self.transform(image=img, bboxes=bboxes)
            transformed_image = transformed['image'] # (3, 416, 416)
            transformed_bboxes = transformed['bboxes'] # (n_objects, 5) x,y,w,h,label

        gt_boxes = torch.empty(0, 4)
        gt_labels = torch.empty(0, dtype=torch.uint8)

        for bbox in transformed_bboxes:
            x, y, w, h, label = bbox
            gt_boxes = torch.vstack([gt_boxes, torch.tensor([x, y, w, h], dtype=torch.float32)])
            gt_labels = torch.hstack([gt_labels, torch.tensor(label, dtype=torch.uint8)])

        return transformed_image, gt_boxes, gt_labels

