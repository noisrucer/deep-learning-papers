import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset
from utils import(
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_list_boxes,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
)

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 5e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {DEVICE}")
BATCH_SIZE = 32
WEIGHT_DECAY = 0
EPOCHS = 3000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

max_map = 0
save_map = False
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = YOLOv1(grid_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=DEVICE), model, optimizer)

    train_dataset = VOCDataset(
        "data/8examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, confidence_threshold=0.4, device=DEVICE
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        print(f"Train mAP: {mean_avg_prec}")
        global max_map
        global save_map
        if mean_avg_prec > 0.99 and not save_map:
            checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    }
            save_checkpoint(checkpoint, fname=LOAD_MODEL_FILE)
            max_map = mean_avg_prec
            save_map = True
        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()
