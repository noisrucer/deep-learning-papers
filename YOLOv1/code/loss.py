import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, box_format="midpoint"):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.box_format = box_format
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.mse = nn.MSELoss(reduction="sum") # according to the paper

    def forward(self, preds, labels):
        '''
        Returns the loss of YOLOv1

        Parameters:
            preds (tensor): predicted bounding boxes in the shape
                            (batch_size, S, S, 30)
            labels (tensor): ground truth bounding boexes in the shape
                            (batch_size, S, S, 30)

        Returns:
            loss (float): The final loss consists of
                          1. Coordinate Loss
                          2. Confidence Loss(obj/noobj)
                          3. Class Loss
        0:20 - class label one-hot vector
        20 - box1 confidence
        21:25 - box1 (x,y,w,h)
        25 - box2 confidence
        26:30 - box2 (x,y,w,h)
        '''

        '''
        First, we need to determine which box is responsible for detecting the
        obj in a specific grid cell, given that an object exists. As stated in
        the original paper, only ONE predicted bounding box should be responsible.
        This is also a limitation of YOLOv1. The way to determine the responsibility
        is to compare both predictions' IoU with the ground truth box and pick
        the one with the highest IoU, given an object exists.
        '''
        preds = preds.reshape(-1, self.S, self.S, self.C + 5 * self.B)

        # Ground truth coordinates, class one-hot vector, confidence
        gt_coord = labels[..., 21:25]
        gt_class = labels[..., 0:20]
        gt_confidence = labels[..., 20:21]

        # Same as confidence..but denote Identity
        Iobj = labels[..., 20:21]

        # COORDINATES for box 1, 2
        box1_coord = preds[..., 21:25]
        box2_coord = preds[..., 26:30]

        # CLASS LABEL one-hot vector
        pred_class = preds[..., 0:20]

        # CONFIDENCE for box 1,2
        box1_confidence = preds[..., 20:21]
        box2_confidence = preds[..., 25:26]

        # IoU score for box 1,2
        box1_iou = intersection_over_union(
            box1_coord,
            gt_coord,
            box_format=self.box_format
        )

        box2_iou = intersection_over_union(
            box2_coord,
            gt_coord,
            box_format=self.box_format
        )

        iou_combined = torch.cat(
            (box1_iou, box2_iou),
            dim = -1
        )

        # select best box with higher IoU
        # (batch_size, S, S, 1) -> 0 or 1
        best_box_num = iou_combined.argmax(
            dim = -1, keepdim=True
        )


        # BEST box confidence
        best_box_confidence = (
            (1 - best_box_num) * box1_confidence
            + best_box_num * box2_confidence
        )

        # BEST box coordinates (x,y,w,h)
        # (batch size, S, S, 4)
        best_box_coord = (
            (1 - best_box_num) * box1_coord # if 0
            + best_box_num * box2_coord  # if 1
        )


        ##############################
        #      COORDINATE LOSS       #
        ##############################
        torch.autograd.set_detect_anomaly(True)
        best_box_coord[...,2:4] = torch.sign(best_box_coord[...,2:4]) * torch.sqrt(
            torch.abs(best_box_coord[...,2:4] + 1e-6)
        )

        gt_coord[...,2:4] = torch.sqrt(gt_coord[...,2:4])

        coord_loss = self.lambda_coord * self.mse(
            torch.flatten(Iobj * best_box_coord, end_dim=-2),
            torch.flatten(Iobj * gt_coord, end_dim=-2)
        )


        ##############################
        #      CONFIDENCE LOSS       #
        ##############################
        # If YES object
        obj_confidence_loss = self.mse(
            torch.flatten(Iobj * best_box_confidence, end_dim=-2),
            torch.flatten(Iobj * gt_confidence,end_dim=-2)
        )

        # If NO object
        noobj_confidence_loss = self.mse(
            torch.flatten((1 - Iobj) * box1_confidence, end_dim=-2),
            torch.flatten((1 - Iobj) * gt_confidence, end_dim=-2)
        )

        noobj_confidence_loss += self.mse(
            torch.flatten((1 - Iobj) * box2_confidence, end_dim=-2),
            torch.flatten((1 - Iobj) * gt_confidence, end_dim=-2)
        )

        confidence_loss = (
            obj_confidence_loss
            + self.lambda_noobj * noobj_confidence_loss
        )

        ##############################
        #         CLASS LOSS         #
        ##############################

        class_loss = self.mse(
            torch.flatten(Iobj * pred_class, end_dim=-2),
            torch.flatten(Iobj * gt_class, end_dim=-2)
        )

        return coord_loss + confidence_loss + class_loss
