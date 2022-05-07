import torch
from dataset import VOCDataset
from torchvision import transforms
from plot_image import plot_bbox
import time

LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_NUMBER = 1

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

dataset = VOCDataset(
    'data/100examples.csv',
    transform=transform,
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR
)

image_test = dataset[8][0]
plot_bbox(image_test)
#  plot_bbox(dataset[0][0])
#  plot_bbox(dataset[2][0])
#  plot_bbox(dataset[3][0])

