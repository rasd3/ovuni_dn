import os
import pickle as pkl
import numpy as np

# 1) pickle 파일 로드
pkl_path = 'sunrgbd_infos_train_pls_ens_10c36c.pkl'
with open(pkl_path, 'rb') as f:
    data = pkl.load(f)

# 2) 클래스 목록 정의
allowed_classes = [
    'chair', 'table', 'pillow', 'sofa_chair', 'desk', 'bed', 'sofa',
    'computer', 'box', 'lamp', 'garbage_bin', 'cabinet', 'shelf', 'drawer',
    'sink', 'night_stand', 'kitchen_counter', 'paper', 'end_table',
    'kitchen_cabinet', 'picture', 'book', 'stool', 'coffee_table',
    'bookshelf', 'painting', 'key_board', 'dresser', 'tv', 'whiteboard',
    'cpu', 'toilet', 'file_cabinet', 'bench', 'ottoman', 'plant', 'monitor',
    'printer', 'recycle_bin', 'door', 'fridge', 'towel', 'cup', 'mirror',
    'laptop', 'cloth'
]

# 2-1) 클래스 매핑 생성
class_to_index = {cls: idx for idx, cls in enumerate(allowed_classes)}

# 3) 텍스트 라벨 파싱 함수
# 3) 텍스트 라벨 파싱 함수
def parse_label_file(label_file_path):
    """
    한 줄 예시:
    class_name xmin ymin dx dy cx cy cz length width height heading_x heading_y
    """
    annotations = []
    if not os.path.exists(label_file_path):
        return annotations

    with open(label_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split()
            class_name = parts[0]
            data = [float(x) for x in parts[1:]]

            xmin = data[0]
            ymin = data[1]
            dx = data[2]
            dy = data[3]
            cx, cy, cz = data[4:7]
            width, length, height = data[7:10]
            heading_x, heading_y = data[10:12]

            # 2D bounding box
            xmax = xmin + dx
            ymax = ymin + dy
            bbox_2d = [xmin, ymin, xmax, ymax]

            # 3D bounding box
            size = [length * 2, width * 2, height * 2]
            heading_angle = np.arctan2(heading_y, heading_x)
            bbox_3d = [cx, cy, cz] + size + [heading_angle]

            ann = {
                'name': class_name,
                'bbox_2d': bbox_2d,
                'bbox_3d': bbox_3d
            }
            annotations.append(ann)
    return annotations

# 4) 중복 판단 함수
def is_same_3d_box(box_a, box_b, tol=1e-4):
    """
    box_a, box_b 형식: [cx, cy, cz, dx, dy, dz, heading]
    두 박스가 거의 동일한지 판단(오차 범위 tol 사용).
    """
    box_a = np.array(box_a, dtype=float)
    box_b = np.array(box_b, dtype=float)
    diff = np.abs(box_a - box_b)
    return np.all(diff < tol)

def remove_duplicates_from_pkl_annos(old_annos, label_anns):
    """
    old_annos: pkl에 저장된 annos (dict)
    label_anns: label/에서 파싱된 ann 목록 (list of dict: {'name', 'bbox_2d', 'bbox_3d'})
    -> label/와 겹치는 3D 박스를 old_annos에서 제거해서 반환
    """
    if 'gt_boxes_upright_depth' not in old_annos:
        return old_annos
    old_boxes_3d = old_annos['gt_boxes_upright_depth']
    old_names = old_annos['name']
    old_bboxes_2d = old_annos['bbox']
    old_scores = old_annos.get('score', None)

    keep_indices = []
    for i, old_box in enumerate(old_boxes_3d):
        # label_anns의 모든 3D 박스와 비교
        is_duplicate = False
        for ann in label_anns:
            if is_same_3d_box(old_box, ann['bbox_3d']):
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)

    # keep_indices에 해당하는 요소만 남김
    old_boxes_3d = old_boxes_3d[keep_indices]
    old_names = old_names[keep_indices]
    old_bboxes_2d = old_bboxes_2d[keep_indices]
    if old_scores is not None:
        old_scores = old_scores[keep_indices]

    # annos 갱신
    old_annos['gt_boxes_upright_depth'] = old_boxes_3d
    old_annos['name'] = old_names
    old_annos['bbox'] = old_bboxes_2d
    if old_scores is not None:
        old_annos['score'] = old_scores
    old_annos['gt_num'] = len(keep_indices)

    return old_annos


# 5) 전체 데이터 순회하며 라벨 업데이트
for i, info in enumerate(data):
    print(i)
    idx = info['image']['image_idx']  # 예: 5051
    
    # (a) pkl의 annos (old_annos)와 label/의 라벨( label_anns ) 읽기
    old_annos = info['annos']
    label_path = f'/shared/workspace/ovuni_dn/data/sunrgbd/sunrgbd_trainval/label/{idx:06d}.txt'
    label_anns = parse_label_file(label_path)

    # (b) pkl annos에서 label/와 겹치는 박스 제거
    old_annos = remove_duplicates_from_pkl_annos(old_annos, label_anns)

    # (c) label_v2 라벨 읽기
    label_v2_path = f'/shared/workspace/ovuni_dn/data/sunrgbd/sunrgbd_trainval/label_v1/{idx:06d}.txt'
    label_v2_anns = parse_label_file(label_v2_path)

    # (d) 이제 old_annos(= pkl에서 label/와 겹치는 것 뺀 결과)에 label_v2 라벨을 합친다
    merged_anns = []
    # 1) 먼저 pkl annos 쪽 정보(= old_annos)를 list(dict) 형태로 변환
    if 'name' in old_annos:
        for name, box_3d, box_2d in zip(old_annos['name'],
                                        old_annos['gt_boxes_upright_depth'],
                                        old_annos['bbox']):
            merged_anns.append({
                'name': name,
                'bbox_3d': box_3d,
                'bbox_2d': box_2d
            })

    # 2) label_v2에서 가져온 항목을 merged_anns에 추가
    base_num = len(merged_anns)
    merged_anns += label_v2_anns

    # (e) 필요한 클래스만 남기기
    final_anns = []
    for idx, ann in enumerate(merged_anns):
        if idx < base_num or ann['name'] in allowed_classes[:10]:
            final_anns.append(ann)

    # (f) 최종 annos 갱신: old_annos를 덮어쓰기
    new_names = []
    new_classes = []
    new_boxes_3d = []
    new_bboxes_2d = []
    for ann in final_anns:
        new_names.append(ann['name'])
        new_classes.append(class_to_index[ann['name']])
        new_boxes_3d.append(ann['bbox_3d'])
        new_bboxes_2d.append(ann['bbox_2d'])

    new_boxes_3d = np.array(new_boxes_3d, dtype=np.float32)
    new_bboxes_2d = np.array(new_bboxes_2d, dtype=np.float32)

    info['annos']['name'] = np.array(new_names)
    info['annos']['class'] = np.array(new_classes, dtype=np.int32)
    info['annos']['gt_boxes_upright_depth'] = new_boxes_3d
    info['annos']['bbox'] = new_bboxes_2d
    info['annos']['gt_num'] = len(new_names)

# 6) 수정된 data를 다시 저장
output_pkl_path = 'sunrgbd_infos_train_pls_ens_10c36c_updated.pkl'
with open(output_pkl_path, 'wb') as f:
    pkl.dump(data, f)
