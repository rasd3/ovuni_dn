
import sys
import os
import copy
from tqdm import tqdm
import torch
import numpy as np
import pickle as pkl
from shapely.geometry import Polygon

def box_iou_3d(box1, box2):
    def get_rotated_corners(box):
        cx, cy, cz, dx, dy, dz, theta = box
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        corners = np.array([
            [-dx / 2, -dy / 2],
            [ dx / 2, -dy / 2],
            [ dx / 2,  dy / 2],
            [-dx / 2,  dy / 2],
        ])
        rotated_corners = np.dot(corners, rot_matrix.T) + np.array([cx, cy])
        return rotated_corners

    def compute_inter_area(box1, box2):
        def validate_polygon(polygon):
            if not polygon.is_valid:
                return polygon.buffer(0)  # 자가 교차 문제 등을 해결
            return polygon
        corners1 = get_rotated_corners(box1)
        corners2 = get_rotated_corners(box2)
        
        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)
        
        intersection = poly1.intersection(poly2)
        
        # 교차 면적 반환
        return intersection.area if not intersection.is_empty else 0

    def compute_inter_height(box1, box2):
        z1_min = box1[2] - box1[5] / 2
        z1_max = box1[2] + box1[5] / 2
        z2_min = box2[2] - box2[5] / 2
        z2_max = box2[2] + box2[5] / 2
        
        # 교차 높이 계산
        inter_z_min = max(z1_min, z2_min)
        inter_z_max = min(z1_max, z2_max)
    
        return max(0, inter_z_max - inter_z_min)

    #  diff_iou_rotated_3d(torch.tensor([box1]), torch.tensor([box2]))
    inter_area = compute_inter_area(box1, box2)
    inter_height = compute_inter_height(box1, box2)
    inter_volume = inter_area * inter_height
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]
    
    union_volume = vol1 + vol2 - inter_volume
    
    iou = inter_volume / union_volume
    return iou

if len(sys.argv) < 3:
    raise ValueError(
        "\n[Error] Two pkl file paths are required.\n"
        "Usage: python script.py <preds.pkl> <o_pkl.pkl>\n"
        "Example: python script.py test.pkl sunrgbd_infos_val_46cls.pkl\n"
    )

preds_path = sys.argv[1]
o_pkl_path = sys.argv[2]

preds = pkl.load(open(preds_path, 'rb'))
o_pkl = pkl.load(open(o_pkl_path, 'rb'))
refs = pkl.load(open('sunrgbd_infos_train_46cls.pkl', 'rb'))
o_preds = copy.deepcopy(preds)

assert len(preds) == len(o_pkl)
assert 'annos' in o_pkl[0]

CONF_THR = 0.5
ERASE_E_NOVEL = False
ERASE_OVL_NOVEL = False
NUM_BASE_CLASS = 10
IOU_THRESHOLD = 0.2  # IoU Threshold for exclusion

ref_pkl = []
for idx in tqdm(range(len(preds)), desc='refinement train pkl'):
    pred, o_gt = preds[idx], o_pkl[idx]
    ref = refs[idx]
    
    # preds filtering
    pred['boxes_3d'] = pred['boxes_3d'].tensor
    pred['boxes_3d'][:, 2] += pred['boxes_3d'][:, 5] / 2
    mask = (pred['scores_3d'] > CONF_THR) & (pred['labels_3d'] >= NUM_BASE_CLASS)
    pred['scores_3d'] = pred['scores_3d'][mask]
    pred['boxes_3d'] = pred['boxes_3d'][mask]
    pred['labels_3d'] = pred['labels_3d'][mask]

    # erase novel class
    if ERASE_E_NOVEL and o_gt['annos']['gt_num']:
        mask = o_gt['annos']['class'] < 0
        o_gt['annos']['gt_num'] = mask.sum()
        o_gt['annos']['name'] = o_gt['annos']['name'][mask]
        o_gt['annos']['gt_boxes_upright_depth'] = o_gt['annos']['gt_boxes_upright_depth'][mask]
        o_gt['annos']['class'] = o_gt['annos']['class'][mask]

    # IoU Filtering
    if o_gt['annos']['gt_num'] == 0:
        o_gt['annos']['gt_boxes_upright_depth'] = np.zeros((0, 7), dtype=np.float32)
        o_gt['annos']['class'] = np.zeros((0), dtype=np.int64)
        o_gt['annos']['bbox'] = np.zeros((0, 4), dtype=np.int64)
    if False:
        filtered_boxes = []
        filtered_labels = []

        for i in range(pred['boxes_3d'].shape[0]):
            keep = True
            for gt_box in o_gt['annos']['gt_boxes_upright_depth']:
                iou = box_iou_3d(pred['boxes_3d'][i].cpu().numpy(), gt_box)
                if iou >= IOU_THRESHOLD:
                    keep = False
                    break
            if keep:
                filtered_boxes.append(pred['boxes_3d'][i].cpu().numpy())
                filtered_labels.append(pred['labels_3d'][i].cpu().numpy())

        # Stack each part to o_gt['annos']
        if len(filtered_boxes) > 0:
            filtered_boxes = np.array(filtered_boxes, dtype=np.float32).reshape(-1, 7)
            filtered_labels = np.array(filtered_labels, dtype=np.int64)
            o_gt['annos']['gt_boxes_upright_depth'] = np.vstack((o_gt['annos']['gt_boxes_upright_depth'], filtered_boxes))
            o_gt['annos']['class'] = np.hstack((o_gt['annos']['class'], filtered_labels))
            o_gt['annos']['gt_num'] = len(o_gt['annos']['class'])
    else:
        # IoU Filtering for o_gt (Exclude overlapping ground-truth boxes)
        filtered_gt_boxes = []
        filtered_gt_labels = []

        for i in range(o_gt['annos']['gt_boxes_upright_depth'].shape[0]):
            keep = True
            for pred_box in pred['boxes_3d']:
                iou = box_iou_3d(o_gt['annos']['gt_boxes_upright_depth'][i], pred_box.cpu().numpy())
                if iou >= IOU_THRESHOLD:
                    keep = False
                    break
            if keep:
                filtered_gt_boxes.append(o_gt['annos']['gt_boxes_upright_depth'][i])
                filtered_gt_labels.append(o_gt['annos']['class'][i])

        # Add filtered_gt_boxes back to o_gt
        filtered_gt_boxes = np.array(filtered_gt_boxes, dtype=np.float32).reshape(-1, 7) if filtered_gt_boxes else np.zeros((0, 7), dtype=np.float32)
        filtered_gt_labels = np.array(filtered_gt_labels, dtype=np.int64) if filtered_gt_labels else np.zeros((0,), dtype=np.int64)

        # Combine with pred_boxes and pred_labels
        pred_boxes = pred['boxes_3d'].cpu().numpy()
        pred_labels = pred['labels_3d'].cpu().numpy()

        combined_boxes = np.vstack((filtered_gt_boxes, pred_boxes)) if pred_boxes.size > 0 else filtered_gt_boxes
        combined_labels = np.hstack((filtered_gt_labels, pred_labels)) if pred_labels.size > 0 else filtered_gt_labels

        # Update o_gt annotations
        o_gt['annos']['gt_boxes_upright_depth'] = combined_boxes
        o_gt['annos']['class'] = combined_labels
        o_gt['annos']['gt_num'] = len(combined_labels)

    ref_pkl.append(o_gt)

# Save to a new pickle file
preds_dir = os.path.dirname(preds_path)  # Directory of the first pkl file
o_pkl_filename = os.path.basename(o_pkl_path)  # Get the filename of the second pkl
refined_filename = f"{os.path.splitext(o_pkl_filename)[0]}_refined_cat.pkl"  # Add '_refined' suffix
output_path = os.path.join(preds_dir, refined_filename)  # Final output path

# Save to the dynamically created path
with open(output_path, 'wb') as f:
    pkl.dump(ref_pkl, f)

print(f"Refined annotations have been saved to {output_path}")
