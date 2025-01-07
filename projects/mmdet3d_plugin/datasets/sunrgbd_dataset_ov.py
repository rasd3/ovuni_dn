# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from ..core.indoor_eval import indoor_eval_ov, indoor_eval_ov_pklpkl
from ..core.indoor_eval_cls_agn import indoor_eval_ov_cls_agn, indoor_eval_ov_cls_agn_yc
from ..core.indoor_eval_cls_agn2 import indoor_eval_ov_cls_agn


import mmdet3d
#from mmdet.datasets import DATASETS
from mmdet3d.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets import SUNRGBDDataset

__mmdet3d_version__ = float(mmdet3d.__version__[:3])

@DATASETS.register_module()
class SUNRGBDDataset_OV(SUNRGBDDataset):
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 seen_classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 cls_agnostic=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        self.seen_classes = seen_classes
        self.classes = seen_classes
        self.cls_agnostic = cls_agnostic
    
    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 iou_thr_2d=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 axis_aligned_lw=False):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            iou_thr (list[float], optional): AP IoU thresholds for 3D
                evaluation. Default: (0.25, 0.5).
            iou_thr_2d (list[float], optional): AP IoU thresholds for 2D
                evaluation. Default: (0.5, ).
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        if self.cls_agnostic:
            ret_dict = indoor_eval_ov(
                self.seen_classes,
                gt_annos,
                results,
                iou_thr,
                label2cat,
                logger=logger,
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d,
                axis_aligned_lw=axis_aligned_lw)
        else:
            ret_dict = indoor_eval_ov(
                self.seen_classes,
                gt_annos,
                results,
                iou_thr,
                label2cat,
                logger=logger,
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d,
                axis_aligned_lw=axis_aligned_lw)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict


@DATASETS.register_module()
class SUNRGBDDataset_OV_pklpkl(SUNRGBDDataset):
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pl_ann_file,
                 pipeline=None,
                 classes=None,
                 seen_classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        self.seen_classes = seen_classes
        self.classes = seen_classes

        self.data_infos_pl = self.load_annotations(pl_ann_file)

    def convert_pl_to_det(self, pl_annos):
        for idx in range(len(pl_annos)):
            if 'annos' in pl_annos[idx]:
                if pl_annos[idx]['annos']['gt_num'] == 0:
                    pl_annos[idx]['labels_3d'] = torch.tensor([], dtype=torch.int64)
                    pl_annos[idx]['boxes_3d'] = torch.tensor([], dtype=torch.float32).reshape(0, 7)
                    pl_annos[idx]['scores_3d'] = torch.tensor([], dtype=torch.float32)
                else:
                    pl_annos[idx]['labels_3d'] = torch.tensor(pl_annos[idx]['annos']['class'], dtype=torch.int64)
                    pl_annos[idx]['boxes_3d'] = torch.tensor(pl_annos[idx]['annos']['gt_boxes_upright_depth'], dtype=torch.float32)
                    pl_annos[idx]['scores_3d'] = torch.ones_like(pl_annos[idx]['labels_3d'], dtype=torch.float32)
            else:
                if pl_annos[idx]['gt_num'] == 0:
                    pl_annos[idx]['labels_3d'] = torch.tensor([], dtype=torch.int64)
                    pl_annos[idx]['boxes_3d'] = torch.tensor([], dtype=torch.float32).reshape(0, 7)
                    pl_annos[idx]['scores_3d'] = torch.tensor([], dtype=torch.float32)
                else:
                    pl_annos[idx]['labels_3d'] = torch.tensor(pl_annos[idx]['class'], dtype=torch.int64)
                    pl_annos[idx]['boxes_3d'] = torch.tensor(pl_annos[idx]['gt_boxes_upright_depth'], dtype=torch.float32)
                    pl_annos[idx]['scores_3d'] = torch.ones_like(pl_annos[idx]['labels_3d'], dtype=torch.float32)
        return pl_annos
    
    def evaluate(self,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 iou_thr_2d=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 axis_aligned_lw=False):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            iou_thr (list[float], optional): AP IoU thresholds for 3D
                evaluation. Default: (0.25, 0.5).
            iou_thr_2d (list[float], optional): AP IoU thresholds for 2D
                evaluation. Default: (0.5, ).
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        assert len(self.data_infos_pl) == len(self.data_infos)
        gt_annos = [info['annos'] for info in self.data_infos]
        pl_annos = [info['annos'] for info in self.data_infos_pl]
        pl_annos_cvt = self.convert_pl_to_det(pl_annos)
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval_ov_pklpkl(
            self.seen_classes,
            gt_annos,
            pl_annos,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            axis_aligned_lw=axis_aligned_lw)

        return ret_dict
