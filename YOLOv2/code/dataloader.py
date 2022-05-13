import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomDataset

class CustomDataLoader(DataLoader):
    def __init__(self, data_dir='./data/train', label_dir='./data/labels',
                 mode='train', batch_size=8, shuffle=True, drop_last=False, num_workers=2):
        transform = get_transform(mode=mode)

        dataset = CustomDataset(data_dir=data_dir, label_dir=label_dir, transform=transform)

        init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'drop_last': drop_last,
            'num_workers': num_workers
        }

        super().__init__(**init_kwargs)


def get_transform(mode='train'):
    if mode == 'train':
        transform = A.Compose([
            A.Resize(416, 416),
            A.Normalize(mean=[0,0,0], std=[1,1,1]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo'))
    elif mode == 'val':
        transform = A.Compose([
            A.Resize(416, 416),
            A.Normalize(mean=[0,0,0], std=[1,1,1]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo'))
    else:
        transform = A.Compose([
            A.Resize(416, 416),
            A.Normalize(mean=[0,0,0], std=[1,1,1]),
            ToTensorV2()
        ], bbox_params=A.bboxParams(format='yolo'))

    return transform


