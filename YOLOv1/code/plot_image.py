import torch
from dataset import VOCDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from model import YOLOv1
from utils import (
    cellboxes_to_list_boxes,
    non_max_suppression,
)
import time

LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
classes={
    1: 'bicycle',
    6: 'car',
    7: 'cat',
    11: 'dog',
    12: 'horse',
    14: 'peron'
}

color_list = ['red','blue','brown','rosybrown','lightyellow','aquamarine','mediumslateblue','skyblue','darkorchid','purple','cyan','darkcyan','lime','green','lightsteelblue','cornflowerblue','pink','crimson','peru','chocolate']
class_list = list(range(0,20))

colors = dict(zip(class_list,color_list))

def plot_bbox(image_tensor):
    '''
    Draw bounding boxes with image

    Parameters:
        images (tensor): (3, 448, 448)

    Returns
        None
    '''
    start_time = time.time()

    model = YOLOv1(grid_size=7, num_boxes=2, num_classes=20)
    model.load_state_dict(
        torch.load(LOAD_MODEL_FILE, map_location=DEVICE)['model_state_dict']
    )
    # DON'T FORGET!!! training mode might mess up Dropout and BatchNorm
    model.eval()
    predictions = model(image_tensor.unsqueeze(0))

    # 1) Loop through images and predictions
    # 2) Plot image first then the prediction
    pred_boxes = cellboxes_to_list_boxes(predictions)
    pred_boxes_nms = []
    for idx in range(len(pred_boxes)):
        nms_boxes = non_max_suppression(
            pred_boxes[idx],
            iou_threshold=0.5,
            confidence_threshold=0.5,
            box_format="midpoint"
        )
        pred_boxes_nms.append(nms_boxes)

    image = image_tensor.permute(1, 2, 0)
    bboxes = pred_boxes_nms[0]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Convert to np.array to obtain height and width
    image = np.array(image)
    img_h, img_w, _ = image.shape

    ax.imshow(image)

    for box in bboxes:
        box_class, confidence, x, y, w, h = box

        # Need to convert to lower upper x and y
        upper_left_x = x - w / 2
        upper_left_y = y - h / 2
        upper_left_x *= img_w
        upper_left_y *= img_h

        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            width = w * img_w,
            height = h * img_h,
            linewidth=3,
            edgecolor=colors[box_class],
            facecolor='none'
        )

        ax.add_patch(rect)
        ax.text(
            x = upper_left_x,
            y = upper_left_y - 5,
            fontsize=8,
            backgroundcolor = colors[box_class],
            color = 'black',
            s = f"{classes[int(box_class)]}: {confidence:.2f}"
        )
        ax.set_axis_off()

    print(f"Execution Time: {time.time() - start_time} seconds.")
    #  plt.show(block=False)
    plt.axis('off')
    plt.draw()
    plt.pause(5)
    plt.close()
