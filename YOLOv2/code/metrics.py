import torch
import numpy as np
from collections import Counter
from utils import pred_boxes_to_list_boxes, gt_boxes_to_list_boxes, convert_loader_model_to_single_list, intersection_over_union


def mean_average_precision(loader, model, anchors_wh, map_iou_thresh=0.5, iou_thresh=0.5, n_classes=5):
    '''
    Parameters
        pred_boxes (tensor): (B, G, G, 5*(5+n_classes)) B should be ALL batches in the dataloader!
            (t_x, t_y, t_w, t_h, t_o)
        gt_boxes (tensor): (B, n_obj, 4)
        gt_labels (tensor): (B, n_obj)
    '''
    model.eval()

    # Convert predicted boxes and gt boxes into a single list (for each)
    # (train_idx, cls, conf, x, y, w, h)
    all_pred_boxes, all_gt_boxes = convert_loader_model_to_single_list(loader, model, anchors_wh, iou_thresh=0.5)

    average_precisions = []
    print('-'*50 + "Ground Truth" + '-'*50)
    for k, gb in enumerate(all_gt_boxes):
        print("[GT {}] img: {} / class: {}".format(k, gb[0], gb[1]))

    print("-"*100)

    for c in range(n_classes):
        # pred_boxes for current class c
        cls_pred_boxes = [
            box for box in all_pred_boxes
            if box[1] == c]

        # gt_boxes for current class c
        cls_gt_boxes = [
            box for box in all_gt_boxes
            if box[1] == c]

        # If there's no gt box, skip
        if len(cls_gt_boxes) == 0:
            continue

        # Frequency table for each image index
        # This tell how many gt boxes per image
        '''
        {idx 0: x,
         idx 1: x,
         ...}
        '''
        gt_visited = Counter([
            gt[0] for gt in cls_gt_boxes
        ])

        for key, val in gt_visited.items():
            gt_visited[key] = torch.zeros(val)

        # Time to calculate TP/FP
        # First, sort cls_pred_boxes w.r.t confidence score
        cls_pred_boxes.sort(key=lambda box: box[2], reverse=True)
        TP = torch.zeros(len(cls_pred_boxes))
        FP = torch.zeros(len(cls_pred_boxes))
        total_gt_boxes = len(cls_gt_boxes)

        for detection_idx, detection in enumerate(cls_pred_boxes):
            best_iou = 0
            best_iou_gt_idx = None

            # GT boxes for SAME image and SAME class
            same_image_class_gt_boxes = [
                box for box in cls_gt_boxes
                if box[0] == detection[0]
            ]

            for gt_idx, gt in enumerate(same_image_class_gt_boxes):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_iou_gt_idx = gt_idx

            if best_iou > map_iou_thresh:
                print("img: {} / class: {} / best_iou: {} / conf: {}".format(detection[0], c, best_iou, detection[2]))
                # If not visited, then the current predicted detection is "correct"!
                if gt_visited[detection[0]][best_iou_gt_idx] == 0:
                    gt_visited[detection[0]][best_iou_gt_idx] = 1
                    TP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # Now, we found all TP, FP for CURRENT CLASS
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # Precisions, Recalls
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        recalls = TP_cumsum / (total_gt_boxes + 1e-6)

        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = torch.cat([torch.tensor([0]), recalls])

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

