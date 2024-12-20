import cv2
import shutil
import numpy as np
import os
from plyfile import PlyData,PlyElement
#import open3d as o3d
from sklearn.decomposition import PCA
import trimesh
from sklearn.cluster import DBSCAN
type2class_semseg = {
    "toilet": 0,
    "bed": 1,
    "chair": 2,
    "sofa": 3,
    "dresser": 4,
    "table": 5,
    "cabinet": 6,
    "bookshelf": 7,
    "pillow": 8,
    "sink": 9,
    "bathtub": 10,
    "refridgerator": 11,
    "desk": 12,
    "night stand": 13,
    "counter": 14,
    "door":15,
    "curtain": 16,
    "box": 17,
    "lamp": 18,
    "bag": 19
}
class2type_semseg = {
    type2class_semseg[t]: t for t in type2class_semseg
}

def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return


def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)

def heading2rotmat(heading_angle):
    rotmat = np.zeros((3,3))
    rotmat[2,2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
    return rotmat

def convert_from_uvd(u, v, d, intr, pose):
    # u is width index, v is height index
    depth_scale = 1000
    z = d/depth_scale

    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    padding = np.ones_like(u)
    
    uv = np.concatenate([u,v,padding], axis=0)
    xyz = (np.linalg.inv(intr[:3,:3]) @ uv) * np.expand_dims(z,axis=0)
    xyz_local = xyz.copy()
    
    xyz = np.concatenate([xyz,padding], axis=0)



    xyz = pose @ xyz
    xyz[:3,:] /= xyz[3,:] 

    #np.savetxt("xyz.txt", xyz.T, fmt="%.3f")
    return xyz[:3, :].T, xyz_local.T

def get_color_label(xyz, intrinsic_image, rgb, mask):
    mask = mask.transpose(1,2,0)

    height, width, ins_num = mask.shape
    intrinsic_image = intrinsic_image[:3,:3]

    xyz_uniform = xyz/xyz[:,2:3] #最后一个元素归一化成1
    xyz_uniform = xyz_uniform.T

    uv = intrinsic_image @ xyz_uniform #3D映射到2D

    uv /= uv[2:3, :]
    uv = np.around(uv).astype(int)
    uv = uv.T

    uv[:, 0] = np.clip(uv[:, 0], 0, width-1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height-1)

    uv_ind = uv[:, 1]*width + uv[:, 0]
    # import pdb; pdb.set_trace()
    pc_rgb = np.take_along_axis(rgb.reshape([-1,3]), np.expand_dims(uv_ind, axis=1), axis=0)
    pc_ins = np.take_along_axis(mask.reshape([-1,ins_num]), np.expand_dims(uv_ind, axis=1), axis=0).astype(int)
    return pc_rgb, pc_ins

def write_ply(save_path,points,text=True):
    points = [tuple(x) for x in points.tolist()]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def compute_bbox(in_pc):
    pca = PCA(2)
    pca.fit(in_pc[:,:2])
    yaw_vec = pca.components_[0,:]
    yaw = np.arctan2(yaw_vec[1],yaw_vec[0])

    in_pc_tmp = in_pc.copy()
    in_pc_tmp = heading2rotmat(-yaw) @ in_pc_tmp[:,:3].T

    x_min = in_pc_tmp[0,:].min()
    x_max = in_pc_tmp[0,:].max()
    y_min = in_pc_tmp[1,:].min()
    y_max = in_pc_tmp[1,:].max()
    z_min = in_pc_tmp[2,:].min()
    z_max = in_pc_tmp[2,:].max()

    dx = x_max-x_min
    dy = y_max-y_min
    dz = z_max-z_min

    bbox = heading2rotmat(yaw) @ np.array([[x_min,y_min,z_min],[x_max,y_max,z_max]]).T
    bbox = bbox.T
    x_min,y_min,z_min = bbox[0]
    x_max,y_max,z_max = bbox[1]

    cx = (x_min+x_max)/2
    cy = (y_min+y_max)/2
    cz = (z_min+z_max)/2

    rst_bbox = np.expand_dims(np.array([cx, cy, cz, dx/2, dy/2, dz/2, -1*yaw]), axis=0)
    #print(rst_bbox.shape)


    #write_oriented_bbox(rst_bbox, "rst.ply")
    #write_ply(in_pc[:,:3], "pc.ply")
    #print(cx, cy, cz, dx, dy, dz, yaw)
    #exit()
    return rst_bbox

def get_split_point(labels):
    index = np.argsort(labels)
    label = labels[index]
    label_shift = label.copy()
    
    label_shift[1:] = label[:-1]
    remain = label - label_shift
    step_index = np.where(remain > 0)[0].tolist()
    step_index.insert(0,0)
    step_index.append(labels.shape[0])
    return step_index,index

def GridSample3D(in_pc, voxel_size):
    in_pc_ = in_pc[:,:3].copy()
    quantized_pc = np.around(in_pc_ / voxel_size)
    quantized_pc -= np.min(quantized_pc, axis=0)
    pc_boundary = np.max(quantized_pc, axis=0) - np.min(quantized_pc, axis=0)
    
    voxel_index = quantized_pc[:,0] * pc_boundary[1] * pc_boundary[2] + quantized_pc[:,1] * pc_boundary[2] + quantized_pc[:,2]
    
    split_point, index = get_split_point(voxel_index)
    
    in_pc = in_pc[index,:]
    out_pc = in_pc[split_point[:-1],:]
        
    #remap index in_pc to out_pc
    remap = np.zeros(in_pc.shape[0])
        
    for ind in range(len(split_point)-1):
        cur_start = split_point[ind]
        cur_end = split_point[ind+1]
        remap[cur_start:cur_end] = ind
    
    remap_back = remap.copy()
    remap_back[index] = remap
    
    remap_back = remap_back.astype(int64)
    return out_pc, remap_back

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """Input is NxC, output is num_samplexC"""
    if replace is None:
        replace = pc.shape[0] < num_sample
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def downsample_pc(pc):
    rst_pc, _ = GridSample3D(pc, 0.013)

    if rst_pc.shape[0] < 20000:
        rst_pc = random_sampling(pc, 30000)

    if rst_pc.shape[0] > 100000:
        rst_pc = random_sampling(rst_pc, 100000)

    return rst_pc

def make_pseudo_label(masks, gt, scores=None, classes=None, annos=None):

    bbox_num = masks.shape[0]

    color_map = cv2.imread(os.path.join('../../sunrgbd/sunrgbd_trainval/', gt['image']['image_path']) )
    color_map = cv2.cvtColor(color_map,cv2.COLOR_BGR2RGB)
    color_map = cv2.resize(color_map, (730, 530))
    
    points = np.fromfile(os.path.join('../../sunrgbd/points/', str(gt['point_cloud']['lidar_idx']).zfill(6)+'.bin'), dtype=np.float32).reshape(-1, 6)
    xyz = points[:, [0,1,2]]
    
    calib = gt['calib']
    rt_mat = calib['Rt']
    # follow Coord3DMode.convert_point
    rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                        ]) @ rt_mat.transpose(1, 0)
    depth2img = calib['K'] @ rt_mat

    # depth2img = np.array(gt['images']['CAM0']['depth2img'])
    # calib = gt['calib']
    # rt_mat = calib['Rt']
    # rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ rt_mat.transpose(1, 0)
    # depth2img = calib['K'] @ rt_mat
    
    rgb, ins = get_color_label(xyz, depth2img, color_map, masks)

    all_bbox = []
    valid_ins = []
    for mask_ind in range(ins.shape[1]):
        cur_mask = ins[:,mask_ind]
        cur_valid = np.where(cur_mask>0)[0]

        cur_ins_pc = xyz[cur_valid, :].copy()

        step_interval = max((1, int(cur_ins_pc.shape[0]/3000)))
        cur_ins_pc = cur_ins_pc[0:cur_ins_pc.shape[0]:step_interval, :]
        
        if scores[mask_ind] < 0:
            continue

        if cur_ins_pc.shape[0] < 100:
            continue

        db = DBSCAN(eps=0.3, min_samples=100).fit(cur_ins_pc)

        cur_ins_pc_remove_outiler = []
        for cluster in np.unique(db.labels_):
            if cluster < 0:
                continue

            cluster_ind = np.where(db.labels_==cluster)[0]
            if cluster_ind.shape[0] / cur_ins_pc.shape[0] < 0.1 or cluster_ind.shape[0] <= 100:
                continue

            cur_ins_pc_remove_outiler.append(cur_ins_pc[cluster_ind,:])


        if len(cur_ins_pc_remove_outiler) < 1:
            continue

        valid_ins.append(mask_ind)
        cur_ins_pc = np.concatenate(cur_ins_pc_remove_outiler, axis=0)

        cur_bbox = compute_bbox(cur_ins_pc)
        all_bbox.append(cur_bbox)
        #np.savetxt("debug_pc_%03d.txt"%(mask_ind), cur_ins_pc, fmt="%.3f")

    if len(all_bbox)<1:
        return None, None, None, None
        
    all_bbox = np.concatenate(all_bbox, axis=0)
    all_bbox = np.concatenate([all_bbox, np.expand_dims(classes[valid_ins], axis=1)], axis=1)

    all_bbox[:,3:6] *= 2
    all_bbox[:,6] *= -1

    all_bbox2d = []

    valid_masks = masks[valid_ins]
    valid_scores = scores[valid_ins]
    valid_annos = {
        'segmentation': np.array(annos['segmentation'])[valid_ins].tolist(),
        'score': np.array(annos['score'])[valid_ins].tolist()
    }
    for i in range(len(valid_masks)):
        pmask = valid_masks[i]
        y, x = np.nonzero(pmask)
        bbox2d = [x.min(), y.min(), x.max(), y.max()]
        all_bbox2d.append(bbox2d)

    # return all_bbox, np.array(all_bbox2d)
    return all_bbox, np.array(all_bbox2d), valid_scores, valid_annos
