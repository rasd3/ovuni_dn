import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable
from .indoor_eval import *

# def match_predictions_to_gt_cls(pred, gt):
#     """
#     Match predictions to ground truth based on IoU overlap.

#     Args:
#         pred (dict): Dictionary containing predictions organized as {objectness: {img_id: predicted_bbox}}.
#         gt (dict): Dictionary containing ground truth organized as {classname: {img_id: bbox}}.

#     Returns:
#         dict: Matched predictions organized as {classname: {img_id: predicted_bbox}}.
#     """
#     matched_preds = {}

#     for classname, gt_images in gt.items():
#         matched_preds[classname] = {}
#         for img_id, gt_bboxes in gt_images.items():
#             if img_id in pred:
#                 # Get predicted boxes for the current image and objectness
#                 pred_bboxes = pred[img_id]
#                 ious = []
#                 for pred_bbox, score in pred_bboxes:
#                     max_iou = 0
#                     best_match = None

#                     for gt_bbox in gt_bboxes:
#                         iou = pred_bbox.overlaps(pred_bbox, gt_bbox)
#                         if iou > max_iou:
#                             max_iou = iou
#                             best_match = pred_bbox

#                     if best_match is not None and max_iou > 0:
#                         ious.append((best_match, score))

#                 # Sort the matched boxes by the objectness score
#                 ious = sorted(ious, key=lambda x: x[1], reverse=True)
#                 if ious:
#                     breakpoint()
#                     matched_preds[classname][img_id] = ious[0]

#     return matched_preds

def match_predictions_to_gt_cls(pred, gt):
    """
    Match predictions to ground truth based on IoU overlap.

    Args:
        pred (dict): Dictionary containing predictions organized as {objectness: {img_id: predicted_bbox}}.
        gt (dict): Dictionary containing ground truth organized as {classname: {img_id: bbox}}.

    Returns:
        dict: Matched predictions organized as {classname: {img_id: predicted_bbox}}.
    """
    matched_preds = {}

    for classname, gt_images in gt.items():
        matched_preds[classname] = {}

   # Iterate over each image in the predictions
    for img_id, pred_bboxes in pred.items():
        for pred_bbox, score in pred_bboxes:
            best_match = None
            best_classname = None
            max_iou = 0

            # Iterate over all ground truth classes to find the best match
            for gt_classname, gt_class_images in gt.items():
                if img_id in gt_class_images:
                    for gt_bbox in gt_class_images[img_id]:
                        # Calculate IoU between the predicted box and the ground truth box
                        iou = pred_bbox.overlaps(pred_bbox, gt_bbox)
                        if iou > max_iou:
                            max_iou = iou
                            best_match = pred_bbox
                            best_classname = gt_classname

            # Assign the predicted box to the class with the highest IoU if IoU > 0
            if best_match is not None and max_iou > 0:
                if img_id not in matched_preds[best_classname]:
                    matched_preds[best_classname][img_id] = []
                matched_preds[best_classname][img_id].append((best_match, score))

    return matched_preds

def indoor_eval_ov_cls_agn(
                seen_classes,
                gt_annos,
                dt_annos,
                metric,
                label2cat,
                logger=None,
                box_type_3d=None,
                box_mode_3d=None,
                axis_aligned_lw=False):
    """Indoor Evaluation.

    Evaluate the result of the detection.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Detection annotations. the dict
            includes the following keys

            - labels_3d (torch.Tensor): Labels of boxes.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): \
                3D bounding boxes in Depth coordinate.
            - scores_3d (torch.Tensor): Scores of boxes.
        metric (list[float]): IoU thresholds for computing average precisions.
        label2cat (dict): Map from label to category.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        axis_aligned_lw (bool): whether to use axis-aligned length and width
            to replace the real length and width.
            Default: False.

    Return:
        dict[str, float]: Dict of results.
    """
    assert len(dt_annos) == len(gt_annos) # the number of images should be the same
    pred = {}  # map {class_id: pred}
    gt = {}  # map {class_id: gt}
    for img_id in range(len(dt_annos)):
        # parse detected annotations
        det_anno = dt_annos[img_id]
        if 'pts_bbox' in det_anno.keys():
            det_anno = det_anno['pts_bbox']
        for i in range(len(det_anno['labels_3d'])):
            label = det_anno['labels_3d'].numpy()[i]
            bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
            score = det_anno['scores_3d'].numpy()[i]
            if label not in pred:
                pred[int(label)] = {}
            if img_id not in pred[label]:
                pred[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))

        # parse gt annotations
        gt_anno = gt_annos[img_id]
        if gt_anno['gt_num'] != 0:
            gt_boxes = box_type_3d(
                gt_anno['gt_boxes_upright_depth'],
                box_dim=gt_anno['gt_boxes_upright_depth'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
            # replace the original real length and width
            if axis_aligned_lw:
                corner3d = gt_boxes.corners
                minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
                minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
                minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]
                dims = minmax_box3d[:, 3:] - minmax_box3d[:, :3]
                boxes = gt_anno['gt_boxes_upright_depth'].copy()
                boxes[:, 3:6] = dims
                gt_boxes = box_type_3d(
                    boxes,
                    box_dim=gt_anno['gt_boxes_upright_depth'].shape[-1],
                    origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)

            labels_3d = gt_anno['class']
        else:
            gt_boxes = box_type_3d(np.array([], dtype=np.float32))
            labels_3d = np.array([], dtype=np.int64)

        for i in range(len(labels_3d)):
            label = labels_3d[i]
            bbox = gt_boxes[i]
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    matched_preds = match_predictions_to_gt_cls(pred[1], gt)
    rec, prec, ap = eval_map_recall(matched_preds, gt, metric)
    ret_dict = dict()
    header = ['classes']

    seen_ids = []
    unseen_ids = []
    for i in label2cat.keys():
        if label2cat[i] in seen_classes:
            seen_ids.append(i)
        else:
            unseen_ids.append(i)
    seen_ids = [i for i in seen_ids if i in ap[0].keys()]
    unseen_ids = [i for i in unseen_ids if i in ap[0].keys()]
    if len([ap[0][i] for i in unseen_ids]) > 0 and len([ap[0][i] for i in seen_ids]) > 0:
        seen_aps_25 = np.concatenate([ap[0][i] for i in seen_ids if not np.isnan(ap[0][i])])
        unseen_aps_25 = np.concatenate([ap[0][i] for i in unseen_ids if not np.isnan(ap[0][i])])
        print_log('', logger=logger)
        print_log('seen AP25: ' + str(seen_aps_25.mean()), logger=logger)
        print_log('unseen AP25: ' + str(unseen_aps_25.mean()), logger=logger)
    elif len([ap[0][i] for i in seen_ids]) > 0:
        seen_aps_25 = np.concatenate([ap[0][i] for i in seen_ids])
        print_log('', logger=logger)
        print_log('seen AP25: ' + str(seen_aps_25.mean()), logger=logger)
    else:
        unseen_aps_15 = np.concatenate([ap[0][i] for i in unseen_ids])
        unseen_aps_25 = np.concatenate([ap[1][i] for i in unseen_ids])
        unseen_aps_50 = np.concatenate([ap[2][i] for i in unseen_ids])
        print_log('', logger=logger)
        print_log('unseen AP25: ' + str(unseen_aps_25.mean()), logger=logger)


    table_columns = [[label2cat[label]
                      for label in ap[0].keys()] + ['Overall']]

    for i, iou_thresh in enumerate(metric):
        header.append(f'AP_{iou_thresh:.2f}')
        header.append(f'AR_{iou_thresh:.2f}')
        rec_list = []
        for label in ap[i].keys():
            ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                ap[i][label][0])
        ret_dict[f'mAP_{iou_thresh:.2f}'] = float(np.mean([i for i in list(ap[i].values()) if not np.isnan(i)])) #float(np.mean(list(ap[i].values())))

        table_columns.append(list(map(float, list(ap[i].values()))))
        table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_rec_{iou_thresh:.2f}'] = float(
                rec[i][label][-1])
            rec_list.append(rec[i][label][-1])
        ret_dict[f'mAR_{iou_thresh:.2f}'] = float(np.mean([i for i in rec_list if not np.isnan(i)])) #float(np.mean(rec_list))

        table_columns.append(list(map(float, rec_list)))
        table_columns[-1] += [ret_dict[f'mAR_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict
