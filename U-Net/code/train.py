import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
import cv2

from unet import UNet
from dataset import CarvanaDataset
from utils import check_accuracy, save_predictions

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 10
IMG_RESIZE = (160, 240)
TRAIN_IMG_DIR = 'data/train'
TRAIN_MASK_DIR = 'data/train_masks'
VAL_IMG_DIR = 'data/val'
VAL_MASK_DIR = 'data/val_masks'

def train_epoch(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device=DEVICE)
        y = y.float().unsqueeze(1).to(device=DEVICE)

        optimizer.zero_grad()

        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())


def main():
    train_transform = A.Compose([
        A.Resize(height=IMG_RESIZE[0], width=IMG_RESIZE[1]),
        A.Rotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=IMG_RESIZE[0], width=IMG_RESIZE[1]),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = CarvanaDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        train_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_dataset = CarvanaDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        val_transform
    )

    val_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle = True
    )

    for epoch in range(NUM_EPOCHS):
        train_epoch(train_loader, model, optimizer, loss_fn)
        check_accuracy(train_loader, model, device=DEVICE)
        save_predictions(train_loader, model)


if __name__ == '__main__':
    main()
