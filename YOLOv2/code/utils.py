import torch
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_ious(boxes1, boxes2):
    '''
    Parameters:
        boxes1: (845, 4) corner
        boxes2: (n_obj, 4) corner

    Returns:
        IoUs: (n1, n2)
    '''

    # Intersection
    top_left_xy = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2].unsqueeze(0)) # (n1, n2, 2)
    bottom_right_xy = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:].unsqueeze(0)) # (n1, n2, 2)

    inter_wh = torch.clamp(bottom_right_xy - top_left_xy, min=0) # (n1, n2, 2)
    intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1] # (n1, n2)

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # (n1, )
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # (n2, )

    union = boxes1_area.unsqueeze(1) + boxes2_area.unsqueeze(0) - intersection + 1e-6

    return intersection / union # (n1, n2)


def center2corner(xywh):
    '''
    :param bboxes (B, 4)
    '''
    x1y1 = xywh[..., :2] - xywh[..., 2:] / 2
    x2y2 = xywh[..., :2] + xywh[..., 2:] / 2

    xyxy = torch.cat([x1y1, x2y2], dim=-1)
    return xyxy


def intersection_over_union(bboxes1, bboxes2):
    '''
    Find IoU

    Parameters:
        bboxes1 (tensor): (B, 4) - x,y,w,h
        bboxes1 (tensor): (B, 4) - x,y,w,h

    Returns:
        IoU (tensor): (B, )
    '''

    bboxes1 = center2corner(bboxes1) # (B, xyxy)
    bboxes2 = center2corner(bboxes2) # (B, xyxy)
    top_left_xy = torch.max(bboxes1[..., :2], bboxes2[..., :2]) # (B, 2)
    bottom_right_xy = torch.min(bboxes1[..., 2:], bboxes2[..., 2:]) # (B, 2)

    inter_wh = torch.clamp(bottom_right_xy - top_left_xy, min=0)
    intersection = inter_wh[..., 0] * inter_wh[..., 1]

    bboxes1_area = torch.abs((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]))
    bboxes2_area = torch.abs((bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]))

    union = bboxes1_area + bboxes2_area - intersection + 1e-6

    return intersection / union


def non_max_suppression(bboxes, iou_thresh=0.5, conf_thresh=0.6):
    '''
    Perform non-max suppression

    Parameters:
        bboxes (tensor): (G*G*5, 7) - train_idx, cls, conf, x, y, w, h
        iou_thresh (float): IoU threshold for NMS
        conf_thresh (float): Confidence threshold for NMS
    '''
    bboxes = bboxes.numpy().tolist()

    nms_bboxes = []

    # [1] Remove all bboxes whose confidence < conf_thresh
    bboxes = [bbox for bbox in bboxes if bbox[2] > conf_thresh]

    # [2] Sort bboxes for confidence in descending order
    bboxes.sort(key=lambda bbox: bbox[2], reverse=True)

    # [3] Perform NMS for EACH class
    while(bboxes):
        top_box = bboxes.pop(0)
        nms_bboxes.append(top_box)

        bboxes = [box for box in bboxes
                  if box[1] != top_box[1]
                  or intersection_over_union(
                      torch.tensor(box[3:]),
                      torch.tensor(top_box[3:])
                  ).item() < iou_thresh]

    return nms_bboxes


def convert_loader_model_to_single_list(loader, model, anchors_wh, iou_thresh=0.5, n_classes=5):
    model.eval()
    train_idx = 0

    all_pred_boxes = []
    all_gt_boxes = []

    for batch_idx, (images, gt_boxes, gt_labels) in enumerate(loader):
        images, gt_boxes, gt_labels = images.to(device), gt_boxes.to(device), gt_labels.to(device)

        with torch.no_grad():
            preds = model(images) # (B, 50, 13, 13)

        B = preds.shape[0]
        G = preds.shape[-1]
        preds = torch.permute(preds, (0, 2, 3, 1)) # (B, 13, 13, 50)

        # Combined into a single list (after NMS)
        batch_all_pred_boxes = pred_boxes_to_list_boxes(preds, anchors_wh, train_idx)
        batch_all_gt_boxes = gt_boxes_to_list_boxes(gt_boxes, gt_labels, train_idx)
        train_idx += B

        all_pred_boxes.extend(batch_all_pred_boxes)
        all_gt_boxes.extend(batch_all_gt_boxes)

    model.train()

    return all_pred_boxes, all_gt_boxes


def gt_boxes_to_list_boxes(gt_boxes, gt_labels, train_idx, G=13):
    '''
    Parameters:
        gt_boxes (tensor): (B, n_obj, 4)
        gt_labels (tensor): (B, n_obj)

    Returns:
    batch_all_gt_boxes: List[(train_idx, cls, conf, x, y, w, h)] scaled [0, 13]
    '''
    gt_boxes, gt_labels = gt_boxes.to('cpu'), gt_labels.to('cpu')
    batch_all_gt_boxes = []

    B = gt_boxes.shape[0]

    for b in range(B):
        n_obj = gt_boxes[b].shape[0]
        idx = train_idx + b # train index
        conf = 1.0

        for obj_idx in range(n_obj):
            glabel = gt_labels[b][obj_idx]
            gx, gy, gw, gh = gt_boxes[b][obj_idx] * float(G) # scale [0,13]
            batch_all_gt_boxes.append([idx, glabel, conf, gx, gy, gw, gh])

    return batch_all_gt_boxes



def pred_boxes_to_list_boxes(pred_boxes, anchors_wh, train_idx, n_classes=5):
    '''
    Parameters:
        cell_boxes (tensor): (B, G, G, 5*(5+n_classes))
            (t_x, t_y, t_w, t_h, t_o) + n_classes
        anchors (list): (5, 2) w,h

    Returns:
        list_boxes: (B, # pred boxes, 7) - train_idx, cls, conf, x, y, w, h
            - x, y, w, h scaled 0~13
    '''
    pred_boxes = pred_boxes.to('cpu')
    B = pred_boxes.shape[0]
    G = pred_boxes.shape[1]

    # Convert parameterized prediction to original bbox format: x,y,w,h are scaled [0,13]
    pred_bxby, pred_bwbh, pred_conf, pred_cls = deparametrize_boxes(pred_boxes, anchors_wh)
    pred_bxby = pred_bxby.reshape(B, G*G*5, 2) # (B, # pred boxes, 2)
    pred_bwbh = pred_bxby.reshape(B, G*G*5, 2) # (B, # pred boxes, 2)
    pred_conf = pred_conf.reshape(B, G*G*5).unsqueeze(-1) # (B, # pred boxes, 1)
    pred_cls = pred_cls.reshape(B, G*G*5).unsqueeze(-1) # (B, # pred boxes, 1)
    pred_train_idx = torch.from_numpy(np.arange(train_idx, train_idx+B)).reshape(B,1).expand(B, G*G*5).unsqueeze(-1) # (B, # pred boxes, 1)

    pred_all = torch.cat([
        pred_train_idx, pred_cls, pred_conf, pred_bxby, pred_bwbh
    ], dim=-1) # (B, G*G*5, 7)

    # Perform NMS and convert to one list
    all_pred_boxes = combine_to_single_list(pred_all) # List[(train_idx, cls, conf, x, y, w, h)]

    return all_pred_boxes


def combine_to_single_list(pred_all, iou_thresh=0.5, conf_thresh=0.5):
    '''
    Perform NMS for bboxes for each batch(image) and combine everything into a single list

    Parameters:
        pred_all (tensor): (B, G*G*5, 7)

    Returns
        all_pred_boxes (list): List[(train_idx, cls, conf, x, y, w, h)]
    '''
    all_pred_boxes = []

    B = pred_all.shape[0]

    for b in range(B):
        nms_boxes = non_max_suppression(
            pred_all[b],
            iou_thresh=iou_thresh,
            conf_thresh=conf_thresh
        )

        all_pred_boxes.extend(nms_boxes)

    return all_pred_boxes


def deparametrize_boxes(pred_boxes, anchors_wh, n_classes=5):
    '''
    Parameters:
        pred_boxes (tensor): (B, G, G, 5*(5+n_classes))
            (t_x, t_y, t_w, t_h, t_o) + n_classes
        anchors_wh (list): (5, 2) w,h

    Returns:
        Return de-parametrized predictions

        pred_bxby (tensor): (B, G, G, 5, 2)
        pred_bwbh (tensor): (B, G, G, 5, 2)
        pred_conf (tensor): (B, G, G, 5)
        pred_cls (tensor): (B, G, G, 5)
    '''
    B = pred_boxes.shape[0]
    G = pred_boxes.shape[1]

    pred_boxes = pred_boxes.reshape(-1, G, G, 5, 5 + n_classes) # (B, G, G, 5, 10)

    pred_boxes[..., :2] = pred_boxes[..., :2].sigmoid() # x,y
    pred_boxes[..., 2:4] = pred_boxes[..., 2:4].exp() # w,h
    pred_boxes[..., 4:5] = pred_boxes[..., 4:5].sigmoid() # confidence
    pred_boxes[..., 5:] = F.softmax(pred_boxes[..., 5:], -1) # cls probs

    pred_xy = pred_boxes[..., :2] # (B, G, G, 5, 2)
    pred_wh = pred_boxes[..., 2:4] # (B, G, G, 5, 2)
    pred_conf = pred_boxes[..., 4] # (B, G, G, 5)
    pred_cls = torch.argmax(pred_boxes[..., 5:], dim=-1) # (B, G, G, 5)

    # Compute bx, by
    cx, cy = np.meshgrid(np.arange(G), np.arange(G))
    cxcy = np.concatenate([
        np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1)
    ], axis=-1) # (G, G, 2)

    cxcy = torch.from_numpy(cxcy)
    cxcy = cxcy.reshape(G, G, 1, 2).expand(G, G, 5, 2).type(torch.float32) # (G, G, 5, 2)
    pred_bxby = pred_xy + cxcy # (B, G, G, 5, 2)

    # Compute bw, bh
    anchors_wh = torch.from_numpy(np.array(anchors_wh)) # (5, 2)
    anchors_wh = anchors_wh.reshape(1, 1, 5, 2).expand(G, G, 5, 2).type(torch.float32) # (G, G, 5, 2)
    pred_bwbh = pred_wh * anchors_wh # (B, G, G, 5, 2)

    return pred_bxby, pred_bwbh, pred_conf, pred_cls

def save_checkpoint(state, fname):
    print("=> Saving Checkpoint...")
    torch.save(state, fname)


def load_checkpoint(checkpoint, model, optimizer, lr_scheduler):
    print("=> Loading Checkpoint...")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
