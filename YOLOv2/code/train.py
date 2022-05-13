import torch
import torch.optim as optim

from dataloader import CustomDataLoader
from yolov2 import Yolov2
from loss import Yolo_Loss
from metrics import mean_average_precision
from utils import save_checkpoint, load_checkpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_EPOCHS = 5000
LEARNING_RATE = 5e-6
BATCH_SIZE = 16
LOG_STEP = 1
SAVE_PERIOD = 20

LOAD_MODEL = False
CHECKPOINT_DIR = './saved'
RESUME_PATH = './saved/checkpoint-epoch119.pth.tar'

def train():
    print("Running on {}...".format(device))

    train_loader = CustomDataLoader(
        data_dir='./data/train', label_dir='./data/labels', mode='train',
        batch_size=16, shuffle=True, drop_last=False, num_workers=0
    )

    val_loader = CustomDataLoader(
        data_dir='./data/val', label_dir='./data/labels', mode='val',
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0
    )

    # [2] Model
    model = Yolov2(n_classes=5)
    model = model.to(device)
    model.train()
    anchors = [(2.5221, 3.3145), (3.19275, 4.00944), (4.5587, 4.09892), (5.47112, 7.84053),
                        (6.2364, 8.0071)]

    # [3] Loss Function
    loss_fn = Yolo_Loss()

    # [4] Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # [5] lr_scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    start_epoch = 1
    # LOAD MODEL
    if LOAD_MODEL:
        print("Loading checkpoint...")
        checkpoint = torch.load(RESUME_PATH)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        start_epoch = checkpoint['epoch']

    total_num_batches = len(train_loader)

    for epoch in range(start_epoch, NUM_EPOCHS+1):
        total_loss = 0.0
        for batch_idx, (images, gt_boxes, gt_labels) in enumerate(train_loader):
            images, gt_boxes, gt_labels = images.to(device), gt_boxes.to(device), gt_labels.to(device).type(torch.long)

            optimizer.zero_grad()
            preds = model(images)
            preds = torch.permute(preds, (0, 2, 3, 1))
            loss = loss_fn(preds, gt_boxes, gt_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % LOG_STEP == 0:
                print('Epoch: {} [{}/{}] Loss: {}'.format(
                    epoch,
                    batch_idx+1,
                    total_num_batches,
                    loss.item()
                ))

        lr_scheduler.step()
        #  print("lr:", optimizer.param_groups[0]['lr'])

        total_loss /= total_num_batches

        #  Calculate mAP
        if epoch % 10 == 0:
            print("Evaluating train set...")
            train_mAP = mean_average_precision(train_loader, model, anchors)
            print("Evaluating val set...")
            val_mAP = mean_average_precision(val_loader, model, anchors)
            print("-"*60)
            print("[Epoch {}] train mAP: {} Loss: {}".format(epoch+1, train_mAP, total_loss))
            print("[Epoch {}] val mAP: {}".format(epoch+1, val_mAP))
            print("-"*60, end="\n\n")

        if epoch % SAVE_PERIOD == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler.state_dict()
            }

            print("Saving checkpoint...")
            save_checkpoint(checkpoint, fname=CHECKPOINT_DIR+'/checkpoint-epoch{}.pth.tar'.format(epoch))


if __name__=='__main__':
    train()
