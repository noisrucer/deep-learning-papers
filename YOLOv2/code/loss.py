import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_ious

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Yolo_Loss(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.n_classes = n_classes
        self.anchors = [(2.5221, 3.3145), (3.19275, 4.00944), (4.5587, 4.09892), (5.47112, 7.84053),
                        (6.2364, 8.0071)]


    def forward(self, preds, gt_boxes, gt_labels):
        '''
        Parameters:
            preds: (B, 13, 13, 50)
            gt_boxes: (B, xxx, 4)
            gt_labels: (B, xxx)
        '''

        G = preds.shape[1] # grid size=13

        # (x, y, w, h, confidence, classes)
        preds = preds.view(-1, G, G, 5, 5 + self.n_classes) # (B, G, G, 5, 25)

        preds[..., :2] = preds[..., :2].sigmoid() # x,y
        preds[..., 2:4] = preds[..., 2:4].exp() # w,h
        preds[..., 4:5] = preds[..., 4:5].sigmoid() # conf

        preds_xywh = preds[..., :4] # (B, G, G, 5, 4)
        pred_xy = preds_xywh[..., :2] # (B, G, G, 5, 2)
        pred_wh = preds_xywh[..., 2:] # (B, G, G, 5, 2)
        pred_conf = preds[..., 4] # (B, G, G, 5)
        pred_cls = preds[..., 5:] # (B, G, G, 20)

        resp_mask, gt_xy, gt_wh, gt_conf, gt_cls = self.build_target(gt_boxes, gt_labels, preds_xywh)
        '''
        resp_mask: (B, G, G, 5)
        gt_xy: (B, G, G, 5, 2)
        gt_wh: (B, G, G, 5, 2)
        gt_conf: (B, G, G, 5)
        gt_cls: (B, G, G, 5, 20) one-hot vector
        '''

        # [1] Coordinate XY SSE
        xy_loss = resp_mask.unsqueeze(-1).expand_as(gt_xy) * ((gt_xy - pred_xy.cpu()) ** 2)

        # [2] Coordinate WH Squared SSE
        wh_loss = resp_mask.unsqueeze(-1).expand_as(gt_wh) * ((torch.sqrt(gt_wh) - torch.sqrt(pred_wh.cpu())) ** 2)

        # [3] Obj Confidence Loss
        obj_conf_loss = resp_mask * ((gt_conf - pred_conf.cpu()) ** 2)

        # [4] Noobj Confidence Loss
        no_obj_conf_loss = (1 - resp_mask) * ((gt_conf - pred_conf.cpu()) ** 2)

        # [5] Classification Loss
        pred_cls = F.softmax(pred_cls, dim=-1) # (B, G, G, 20)
        resp_cell = resp_mask.max(-1)[0].unsqueeze(-1).unsqueeze(-1).expand_as(gt_cls) # (B, G, G, 5, 20)
        cls_loss = resp_cell * ((gt_cls - pred_cls.cpu()) ** 2)

        total_loss = (
            5 * xy_loss.sum()
            + 5 * wh_loss.sum()
            + 1 * obj_conf_loss.sum()
            + 0.5 * no_obj_conf_loss.sum()
            + 1 * cls_loss.sum()
        )

        return total_loss


    def build_target(self, gt_boxes, gt_labels, preds_xywh):
        '''
        Parameters:
            gt_boxes: (B, xxx, 4)
            gt_labels: (B, xxx)
            preds_xywh: (B, G, G, 5, 4)
        '''

        B = preds_xywh.shape[0]
        G = preds_xywh.shape[1]

        pred_xy = preds_xywh[..., :2] # (B, G, G, 5, 2)
        pred_wh = preds_xywh[..., 2:] # (B, G, G, 5, 2)

        resp_mask = torch.zeros(B, G, G, 5)
        gt_xy = torch.zeros(B, G, G, 5, 2)
        gt_wh = torch.zeros(B, G, G, 5, 2)
        gt_conf = torch.zeros(B, G, G, 5)
        gt_cls = torch.zeros(B, G, G, 5, self.n_classes)

        center_anchors = self.build_anchors(self.anchors).to(device) # (G, G, 5, 4)
        corner_anchors = self.center2corner(center_anchors).view(G * G * 5, 4).to(device) # (845, 4)

        # Determin "responsible" mask
        for batch_idx in range(B):
            # GT Label
            label = gt_labels[batch_idx] # (n_objs, )

            # GT Boxes
            center_gt_box = gt_boxes[batch_idx] # (n_objs, 4)
            center_gt_box_13 = center_gt_box * float(G)

            corner_gt_box = self.center2corner(center_gt_box)
            corner_gt_box_13 = corner_gt_box * float(G) # (n_objs, 4) 0~13, x,y,w,h

            # IoUs b/w anchors and GT boxes
            iou_anchors_gt = get_ious(corner_anchors, corner_gt_box_13) # (845, n_obj)
            iou_anchors_gt = iou_anchors_gt.view(G, G, 5, -1) # (G, G, 5, n_obj)

            # GT boxes coordinates: 0~1 scale x,y and 0~13 scale w,h
            bxby = center_gt_box_13[..., :2] # (n_obj, 2)
            x_y_ = bxby - bxby.floor() # (n_obj, 2)
            bwbh = center_gt_box_13[..., 2:] # (n_obj, 2)

            n_obj = corner_gt_box.shape[0]

            for obj_idx in range(n_obj):
                cx, cy = bxby[obj_idx]
                cx = int(cx) # cell number
                cy = int(cy) # cell number

                _, max_anchor_idx = iou_anchors_gt[cy, cx, :, obj_idx].max(0) # anchor with maximum IoU with current obj
                j = max_anchor_idx # jth anchor has max IoU with obj_idx'th target box

                resp_mask[batch_idx, cy, cx, j] = 1 # responsible anchor idx
                gt_xy[batch_idx, cy, cx, :] = x_y_[obj_idx]
                w_h_ = bwbh[obj_idx] / torch.FloatTensor(self.anchors[j]).to(device) # b_w / p_w
                gt_wh[batch_idx, cy, cx, j, :] = w_h_
                gt_cls[batch_idx, cy, cx, j, int(label[obj_idx].item())] = 1

            pred_xy_ = pred_xy[batch_idx] # (G, G, 5, 2)
            pred_wh_ = pred_wh[batch_idx] # (G, G, 5, 2)
            center_pred_xy = center_anchors[..., :2].floor() + pred_xy_ # (G, G, 5, 2)
            center_pred_wh = center_anchors[..., 2:] * pred_wh_ # (G, G, 5, 2)
            center_pred_bbox = torch.cat([center_pred_xy, center_pred_wh], dim=-1) # (G, G, 5, 4)
            corner_pred_bbox = self.center2corner(center_pred_bbox).view(-1, 4) # (845, 4)

            iou_pred_gt = get_ious(corner_pred_bbox,corner_gt_box_13) # (845, n_obj)
            iou_pred_gt = iou_pred_gt.view(G, G, 5, -1) # (G, G, 5, n_obj)

            # From the paper, confidence=IoU
            gt_conf[batch_idx] = iou_pred_gt.max(-1)[0] # (G, G, 5)

        return resp_mask, gt_xy, gt_wh, gt_conf, gt_cls


    def build_anchors(self, anchors_wh, grid_size=13):
        '''
        Parameters:
            anchors_wh: base anchors width and hight (5, 2)
            grid_size: 13

        Returns:
            center_anchors: (G, G, 5, 4)
        '''

        G = grid_size

        xx, yy = np.meshgrid(np.arange(G), np.arange(G)) # each (G, G)
        grid_xy = np.concatenate([np.expand_dims(xx, axis=-1), np.expand_dims(yy, axis=-1)], axis=-1) + 0.5 # (G, G, 2)

        grid_xy = torch.from_numpy(grid_xy) # (G, G, 2)
        anchors_wh = torch.from_numpy(np.array(anchors_wh)) # (5, 2)

        grid_xy = grid_xy.view(G, G, 1, 2).expand(G, G, 5, 2).type(torch.float32) # (G, G, 5, 2)
        anchors_wh = anchors_wh.view(1, 1, 5, 2).expand(G, G, 5, 2).type(torch.float32) # (G, G, 5, 2)

        center_anchors = torch.cat([grid_xy, anchors_wh], dim=-1) # (G, G, 5, 4)

        return center_anchors

    def center2corner(self, xywh):
        '''
        Parameters:
            xywh: (G, G, 5, 4) - (cx, cy, w, h)

        Returns:
            xyxy: (G, G, 5, 4) - (xmin, ymin, xmax, ymax)
        '''

        x1y1 = xywh[..., :2] - xywh[..., 2:] / 2
        x2y2 = xywh[..., :2] + xywh[..., 2:] / 2

        xyxy = torch.cat([x1y1, x2y2], dim=-1)
        return xyxy

    def corner2center(self, xyxy):
        wh = xyxy[..., 2:] - xyxy[..., :2]
        xy = (xyxy[..., 2:] + xyxy[..., :2]) / 2
        return torch.cat([xy, wh], dim=-1)
