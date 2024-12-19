import copy
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16
                        
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.builder import build_loss
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d, bbox_overlaps_nearest_3d
from projects.mmdet3d_plugin.core.bbox.util import get_rdiou
from mmdet3d.core.bbox import AxisAlignedBboxOverlaps3D
from mmcv.ops import nms3d, nms_bev
from mmdet3d.core.bbox import Box3DMode, CameraInstance3DBoxes, points_cam2img, DepthInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.models.fusion_layers.coord_transform import coord_2d_transform, apply_3d_transformation
from mmcv.ops import diff_iou_rotated_3d, box_iou_rotated
from projects.mmdet3d_plugin.models.dense_heads.uni3detr_head_clip_dn import MLP, BatchNormDim1Swap, GenericMLP, shift_scale_points, PositionEmbeddingCoordsSine


NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}



@HEADS.register_module()
class Uni3DETRHeadCLIPDNSAF(DETRHead):
    """Head of UVTR. 
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 zeroshot_path=None,
                 multimodal=False,
                 loss_bbox=dict(type='RotatedIoU3DLoss', loss_weight=1.0),
                 loss_iou=dict(type='RotatedIoU3DLoss', loss_weight=1.0),
                 post_processing=None,
                 gt_repeattimes=1,
                 noise_type='jitter',
                 dn_weight=0.5,
                 alpha=0.2,
                 beta=0.45,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1

        super(Uni3DETRHeadCLIPDNSAF, self).__init__(
            *args, transformer=transformer, loss_bbox=loss_bbox, loss_iou=loss_iou, **kwargs)
        
        self.zeroshot_path = zeroshot_path
        zs_weights = np.load(self.zeroshot_path)
        zs_weights = torch.tensor(zs_weights, dtype=torch.float32)
        zs_weights = F.normalize(zs_weights, p=2,dim=1)
        zs_weights = zs_weights.permute(1, 0).contiguous()
        self.register_buffer('opn_zs_weights', zs_weights)
        self.register_buffer('cld_zs_weights', zs_weights[:, :10])
        
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        
        self.fp16_enabled = False
        self.post_processing = post_processing
        self.gt_repeattimes = gt_repeattimes

        self.bbox_noise_scale = 0.3
        self.bbox_noise_trans = 0.
        self.split = 0.75
        self.noise_type = noise_type
        self.dn_weight = dn_weight
        self.num_base_class = 10
        self.alpha = alpha
        self.beta = beta


    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            if _ == 0:
                cls_branch.append(Linear(self.embed_dims, 1024))   
            else:
                cls_branch.append(Linear(1024, 1024)) 
            cls_branch.append(nn.LayerNorm(1024))
            cls_branch.append(nn.ReLU(inplace=True))
        # cls_branch.append(Linear(1024, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        uncertainty_branch = []
        for _ in range(self.num_reg_fcs):
            uncertainty_branch.append(Linear(self.embed_dims, self.embed_dims))
            uncertainty_branch.append(nn.LayerNorm(self.embed_dims))
            uncertainty_branch.append(nn.ReLU(inplace=True))
        uncertainty_branch.append(Linear(self.embed_dims, self.cls_out_channels + 1))
        fc_uncertainty = nn.Sequential(*uncertainty_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        iou_branch = []
        for _ in range(self.num_reg_fcs):
            iou_branch.append(Linear(self.embed_dims, self.embed_dims))
            iou_branch.append(nn.ReLU())
        iou_branch.append(Linear(self.embed_dims, 1))
        iou_branch = nn.Sequential(*iou_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.opn_cls_branches = _get_clones(fc_cls, num_pred)
            self.opn_reg_branches = _get_clones(reg_branch, num_pred)
            self.opn_iou_branches = _get_clones(iou_branch, num_pred)
            self.opn_uncertainty_branches = _get_clones(fc_uncertainty, num_pred)
            self.cld_cls_branches = _get_clones(fc_cls, num_pred)
            self.cld_reg_branches = _get_clones(reg_branch, num_pred)
            self.cld_iou_branches = _get_clones(iou_branch, num_pred)
            self.cld_uncertainty_branches = _get_clones(fc_uncertainty, num_pred)
        else:
            self.opn_cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.opn_reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.opn_iou_branches = nn.ModuleList(
                [iou_branch for _ in range(num_pred)])
            self.cld_cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.cld_reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.cld_iou_branches = nn.ModuleList(
                [iou_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.tgt_embed = nn.Embedding(self.num_query * 2, self.embed_dims)
            self.refpoint_embed = nn.Embedding(self.num_query, 3)
            

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.opn_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
            for m in self.cld_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)


    def prepare_for_dn_cmt(self, batch_size, reference_points, img_metas, points=None):
        if self.training:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(10, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()
            
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                if self.noise_type == 'jitter':
                    rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                    known_bbox_center += torch.mul(rand_prob,
                                                diff) * self.bbox_noise_scale
                    mask = torch.norm(rand_prob, 2, 1) > self.split
                    known_labels[mask] = self.num_classes
                elif self.noise_type == 'ray':
                    box_centers = boxes[:, :3]
                    ray_scales = torch.linspace(0.7, 1.3, groups).view(groups, 1, 1)
                    known_bbox_center = (box_centers.unsqueeze(0) * ray_scales).reshape(-1, 3)

                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padding_bbox_repeated = padding_bbox.unsqueeze(0).repeat(batch_size, 1, 1)
            padded_reference_points = torch.cat([reference_points, padding_bbox_repeated], dim=1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = None

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict


    @auto_fp16(apply_to=("pts_feats",))
    def forward(self, pts_feats, img_metas, fpsbpts, gt_bboxes_3d=None, gt_bboxes=None, points=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        tgt_embed = self.tgt_embed.weight           # nq, 256
        refanchor = self.refpoint_embed.weight      # nq, 6
        #query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

        if fpsbpts is not None:
            bs = fpsbpts.shape[0]

            if pts_feats.requires_grad:
                reference_points = torch.cat([refanchor.unsqueeze(0).expand(bs, -1, -1), inverse_sigmoid(fpsbpts)], 1)
                ref_points, attn_mask, mask_dict = self.prepare_for_dn_cmt(pts_feats.shape[0], reference_points, img_metas, points=points)
                num_dn_q = mask_dict['pad_size']

                tgt_embed = torch.cat([tgt_embed[0:self.num_query], tgt_embed[self.num_query:], tgt_embed[self.num_query:],
                                       tgt_embed[self.num_query:self.num_query+num_dn_q]])

                # for gt dn
                query_embeds = torch.cat([tgt_embed.unsqueeze(0).expand(bs, -1, -1), ref_points], -1)
            else:
                random_point = torch.rand(fpsbpts.shape, device=fpsbpts.device)[:, :self.num_query, :]
                tgt_embed = torch.cat([tgt_embed[0:self.num_query], tgt_embed[self.num_query:], tgt_embed[self.num_query:], tgt_embed[self.num_query:]])
                query_embeds = torch.cat([tgt_embed.unsqueeze(0).expand(bs, -1, -1), torch.cat([refanchor.unsqueeze(0).expand(bs, -1, -1), inverse_sigmoid(fpsbpts), inverse_sigmoid(random_point)], 1)], -1)

        else:
            bs = pts_feats.shape[0]
            tgt_embed = torch.cat([tgt_embed[0:self.num_query] ])
            query_embeds = torch.cat([tgt_embed.unsqueeze(0).expand(bs, -1, -1), torch.cat([refanchor.unsqueeze(0).expand(bs, -1, -1)], 1)], -1)


        # shape: (N, L, C, D, H, W)
        if len(pts_feats.shape) == 5:
            pts_feats = pts_feats.unsqueeze(1)

        hs, init_reference, inter_references = self.transformer(
            pts_feats,
            query_embeds,
            self.num_query,
            reg_branches=self.cld_reg_branches if self.with_box_refine else None,
            img_metas=img_metas,
            ng=4,
        )

        hs = hs.permute(0, 2, 1, 3)

        ret_dict = {}

        # open detection head (no bbox head)
        outputs_classes = []
        outputs_ious = []
        outputs_uncertainties = []

        for lvl in range(len(hs)):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.opn_cls_branches[lvl](hs[lvl])
            outputs_class = torch.matmul(outputs_class, self.opn_zs_weights)

            outputs_uncertainty = self.opn_uncertainty_branches[lvl](hs[lvl])

            outputs_iou = self.opn_iou_branches[lvl](hs[lvl])

            # TODO: check if using sigmoid
            outputs_classes.append(outputs_class)
            outputs_ious.append(outputs_iou)
            outputs_uncertainties.append(outputs_uncertainty)

        outputs_classes = torch.stack(outputs_classes)
        outputs_ious = torch.stack(outputs_ious)
        outputs_uncertainties = torch.stack(outputs_uncertainties)

        opn_outs = {
            'all_cls_scores': outputs_classes,
            'all_iou_preds': outputs_ious,
            'all_uncertainty_preds': outputs_uncertainties,
        }

        if self.training:
            # separate query
            dn_pad_size = mask_dict['pad_size']

            opn_outs.update({
                'all_cls_scores': outputs_classes[:, :, :-dn_pad_size, :],
                'all_iou_preds': outputs_ious[:, :, :-dn_pad_size, :],
                'all_uncertainty_preds': outputs_uncertainties[:, :, :-dn_pad_size, :],
                'dn_cls_scores': outputs_classes[:, :, -dn_pad_size:, :],
                'dn_iou_preds': outputs_ious[:, :, -dn_pad_size:, :],
                'dn_uncertainty_preds': outputs_uncertainties[:, :, -dn_pad_size:, :],
                'dn_mask_dict': mask_dict
            })

        ret_dict.update({'opn_outs': opn_outs})

        # closed detection head
        outputs_classes = []
        outputs_coords = []
        outputs_ious = []
        outputs_uncertainties = []

        for lvl in range(len(hs)):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cld_cls_branches[lvl](hs[lvl])
            if self.training:
                outputs_class = torch.matmul(outputs_class, self.cld_zs_weights)
            else:
                outputs_class = torch.matmul(outputs_class, self.opn_zs_weights)

            outputs_uncertainty = self.cld_uncertainty_branches[lvl](hs[lvl])

            tmp = self.cld_reg_branches[lvl](hs[lvl])
            outputs_iou = self.cld_iou_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3 
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            # transfer to lidar system
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_ious.append(outputs_iou)
            outputs_uncertainties.append(outputs_uncertainty)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_ious = torch.stack(outputs_ious)
        outputs_uncertainties = torch.stack(outputs_uncertainties)

        cld_outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_iou_preds': outputs_ious,
            'all_uncertainty_preds': outputs_uncertainties,
        }

        if self.training:
            # separate query
            dn_pad_size = mask_dict['pad_size']

            cld_outs.update({
                'all_cls_scores': outputs_classes[:, :, :-dn_pad_size, :],
                'all_bbox_preds': outputs_coords[:, :, :-dn_pad_size, :],
                'all_iou_preds': outputs_ious[:, :, :-dn_pad_size, :],
                'all_uncertainty_preds': outputs_uncertainties[:, :, :-dn_pad_size, :],
                'dn_cls_scores': outputs_classes[:, :, -dn_pad_size:, :],
                'dn_bbox_preds': outputs_coords[:, :, -dn_pad_size:, :],
                'dn_iou_preds': outputs_ious[:, :, -dn_pad_size:, :],
                'dn_uncertainty_preds': outputs_uncertainties[:, :, -dn_pad_size:, :],
                'dn_mask_dict': mask_dict
            })

        ret_dict.update({'cld_outs': cld_outs})

        if not self.training:
            total_num_classes = cld_outs['all_cls_scores'][0].shape[-1]  # base + novel + bg
            base_cat_ids = torch.arange(self.num_base_class) 
            base_index = torch.zeros(total_num_classes, dtype=torch.bool, device=cld_outs['all_cls_scores'][0].device)
            base_index[base_cat_ids] = True

            fused_cls_scores = []
            for cld_cls_scores, opn_cls_scores in zip(cld_outs['all_cls_scores'], opn_outs['all_cls_scores']):
                # Confidence fusion using geometric mean
                fused_scores = torch.where(
                    base_index[None, :],
                    cld_cls_scores ** (1 - self.alpha) * opn_cls_scores ** self.alpha,
                    cld_cls_scores ** (self.alpha) * opn_cls_scores ** (1 - self.alpha),
                )
                fused_cls_scores.append(fused_scores)

            cld_outs['all_cls_scores'] = torch.stack(fused_cls_scores)
            return cld_outs

        ret_dict.update({'cld_outs': cld_outs, 'opn_outs': opn_outs})

        return ret_dict

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None,
                           pre='opn_'):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, self.num_query, gt_bboxes_ignore, self.gt_repeattimes)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        if pre == 'opn_':
            labels = gt_bboxes.new_full((num_bboxes, ),
                                        self.num_classes,
                                        dtype=torch.long)
        elif pre == 'cld_':
            labels = gt_bboxes.new_full((num_bboxes, ),
                                        self.num_base_class,
                                        dtype=torch.long)
        else:
            raise NotImplementedError
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_targets = torch.zeros_like(bbox_pred)[..., :7]  #######
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        if pos_inds.numel():
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    pre_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list, pre_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    iou_preds,
                    uncertainty_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    pre='opn_'):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        uncertainty_preds_list = [uncertainty_preds[i] for i in range(num_imgs)]
        pre_list = [pre for _ in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list,
                                           pre_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        uncertainty_preds = torch.cat(uncertainty_preds_list, 0)

        uncertainty_preds = uncertainty_preds[list(range(labels.shape[0])), labels].clip(0.01)
        uncertainty_exp = np.sqrt(2)*torch.exp(-uncertainty_preds[:,None])

        # classification loss
        cls_scores = cls_scores.reshape(-1, cls_scores.shape[-1])
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        bboxes3d = denormalize_bbox(bbox_preds, self.pc_range) 

        iou3d = bbox_overlaps_nearest_3d(bboxes3d, bbox_targets, is_aligned=True, coordinate='depth')
        z1, z2, z3, z4 = self._bbox_to_loss(bboxes3d)[:, 2], self._bbox_to_loss(bboxes3d)[:, 5], self._bbox_to_loss(bbox_targets)[:, 2], self._bbox_to_loss(bbox_targets)[:, 5]
        iou_z = torch.max(torch.min(z2, z4) - torch.max(z1, z3), z1.new_zeros(z1.shape)) / (torch.max(z2, z4) - torch.min(z1, z3) )
        iou3d_dec = (iou3d + iou_z)/2

        loss_cls = self.loss_cls(cls_scores, [labels, iou3d_dec], label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        
        # loss_bbox = self.loss_bbox(bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10] * uncertainty_exp, avg_factor=num_total_pos)

        loss_iou_z = 1 - iou_z[isnotnan]
        loss_iou = self.loss_iou(bboxes3d[isnotnan, :10], bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)
        loss_iou += torch.sum(loss_iou_z * bbox_weights[isnotnan, 0]) / num_total_pos


        iou_preds = iou_preds.reshape(-1)
        iou3d_true = torch.diag(bbox_overlaps_3d(bboxes3d, bbox_targets, coordinate='lidar')).detach()
        loss_iou_pred = torch.sum( F.binary_cross_entropy_with_logits(iou_preds, iou3d_true, reduction='none') * bbox_weights[isnotnan, 0] ) / num_total_pos * 1.2 

        loss_consistency = uncertainty_preds.mean()

        return loss_cls, loss_bbox, loss_iou, loss_iou_pred, loss_consistency

    def dn_loss_single(self,
                       cls_scores,
                       bbox_preds,
                       iou_preds,
                       uncertainty_preds,
                       mask_dict,
                       pre):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        uncertainty_preds_list = [uncertainty_preds[i] for i in range(num_imgs)]
        uncertainty_preds = torch.cat(uncertainty_preds_list, 0)

        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw']
        if pre == 'cld_':
            base_mask = known_labels < self.num_base_class
            known_labels = known_labels[base_mask]
            known_bboxs = known_bboxs[base_mask]
            known_indice = known_indice[base_mask]
            map_known_indice = map_known_indice[base_mask]
            known_labels_raw = known_labels_raw[base_mask]
            bid = bid[base_mask]

        cls_scores = cls_scores[(bid, map_known_indice)]
        bbox_preds = bbox_preds[(bid, map_known_indice)]
        iou_preds = iou_preds[(bid, map_known_indice)]
        num_tgt = known_indice.numel()
        num_total_pos = num_tgt

        # filter task bbox
        task_mask = known_labels_raw != cls_scores.shape[-1]
        task_mask_sum = task_mask.sum()
        
        if task_mask_sum > 0:
            # pred_logits = pred_logits[task_mask]
            # known_labels = known_labels[task_mask]
            bbox_preds = bbox_preds[task_mask]
            known_bboxs = known_bboxs[task_mask]

        bbox_targets = known_bboxs
        labels = known_labels

        uncertainty_preds = uncertainty_preds[list(range(labels.shape[0])), labels].clip(0.01)
        uncertainty_exp = np.sqrt(2)*torch.exp(-uncertainty_preds[:,None])

        # classification loss
        label_weights = torch.ones_like(known_labels)
        cls_scores = cls_scores.reshape(-1, cls_scores.shape[-1])
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split  * self.split
        cls_avg_factor = max(cls_avg_factor, 1)
        #loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        bboxes3d = denormalize_bbox(bbox_preds, self.pc_range) 

        iou3d = bbox_overlaps_nearest_3d(bboxes3d, bbox_targets, is_aligned=True, coordinate='depth')
        # iou3d = box_iou_rotated(DepthInstance3DBoxes(bbox_targets).bev, DepthInstance3DBoxes(bboxes3d).bev, aligned=True)
        z1, z2, z3, z4 = self._bbox_to_loss(bboxes3d)[:, 2], self._bbox_to_loss(bboxes3d)[:, 5], self._bbox_to_loss(bbox_targets)[:, 2], self._bbox_to_loss(bbox_targets)[:, 5]
        iou_z = torch.max(torch.min(z2, z4) - torch.max(z1, z3), z1.new_zeros(z1.shape)) / (torch.max(z2, z4) - torch.min(z1, z3) )
        iou3d_dec = (iou3d + iou_z)/2

        loss_cls = self.loss_cls(cls_scores, [labels, iou3d_dec], label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_weights = torch.ones_like(bbox_preds)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        
        loss_bbox = self.loss_bbox(bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10] * uncertainty_exp, avg_factor=num_total_pos)

        loss_iou_z = 1 - iou_z[isnotnan]
        loss_iou = self.loss_iou(bboxes3d[isnotnan, :10], bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)
        loss_iou += torch.sum(loss_iou_z * bbox_weights[isnotnan, 0]) / num_total_pos

        iou_preds = iou_preds.reshape(-1)
        iou3d_true = torch.diag(bbox_overlaps_3d(bboxes3d, bbox_targets, coordinate='lidar')).detach()
        loss_iou_pred = torch.sum(F.binary_cross_entropy_with_logits(iou_preds, iou3d_true, reduction='none') * bbox_weights[isnotnan, 0] ) / num_total_pos * 1.2 

        loss_consistency = uncertainty_preds.mean()


        return loss_cls*self.dn_weight, loss_bbox*self.dn_weight, loss_iou*self.dn_weight, loss_iou_pred*self.dn_weight, loss_consistency*self.dn_weight
    
    @staticmethod
    def _bbox_to_loss(bbox):
        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)
    
    @staticmethod
    def _loss_to_bbox(bbox):
        return torch.stack(
        ( (bbox[..., 0] + bbox[..., 3]) / 2, (bbox[..., 1] + bbox[..., 4]) / 2, (bbox[..., 2] + bbox[..., 5]) / 2,
            bbox[..., 3] - bbox[..., 0], bbox[..., 4] - bbox[..., 1], bbox[..., 5] - bbox[..., 2], bbox[..., -1] ),
            dim=-1)

    def convert_opn_to_cld(self, labels_list, bboxes_list): 
        filtered_labels_list = []
        filtered_bboxes_list = []

        for img_labels, img_bboxes in zip(labels_list, bboxes_list):
            filtered_labels_per_img = []
            filtered_bboxes_per_img = []

            for labels, bboxes in zip(img_labels, img_bboxes):
                mask = labels < self.num_base_class

                filtered_labels_per_img.append(labels[mask])
                filtered_bboxes_per_img.append(bboxes[mask])

            filtered_labels_list.append(filtered_labels_per_img)
            filtered_bboxes_list.append(filtered_bboxes_per_img)

        return filtered_labels_list, filtered_bboxes_list
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             ret_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        loss_dict = dict()
        all_bbox_preds = ret_dicts['cld_outs']['all_bbox_preds']
        dn_bbox_preds = ret_dicts['cld_outs']['dn_bbox_preds']
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        for pre, preds_dicts in zip(['opn_', 'cld_'], [ret_dicts['opn_outs'], ret_dicts['cld_outs']]):
            print(pre)
            all_cls_scores = preds_dicts['all_cls_scores']
            all_iou_preds = preds_dicts['all_iou_preds']
            all_uncertainty_preds = preds_dicts['all_uncertainty_preds']

            num_dec_layers = len(all_cls_scores)

            all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
            all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
            all_gt_bboxes_ignore_list = [
                gt_bboxes_ignore for _ in range(num_dec_layers)
            ]

            # calculate class and box loss
            if pre == 'cld_':
                all_gt_labels_list, all_gt_bboxes_list = self.convert_opn_to_cld(all_gt_labels_list, all_gt_bboxes_list)
            pre_list = [pre for _ in range(len(all_cls_scores))]
            losses_cls, losses_bbox, losses_iou, losses_iou_pred, losses_consistency = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds, all_iou_preds, all_uncertainty_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list, pre_list)
            if pre == 'opn_':
                losses_bbox = [torch.zeros_like(losses_bbox[0]) for _ in range(len(losses_bbox))]

            # loss from the last decoder layer
            loss_dict[f'{pre}_loss_cls'] = losses_cls[-1]
            if pre == 'cld':
                loss_dict[f'{pre}_loss_bbox'] = losses_bbox[-1]
            loss_dict[f'{pre}_loss_iou'] = losses_iou[-1]
            loss_dict[f'{pre}_loss_iou_pred'] = losses_iou_pred[-1]
            loss_dict[f'{pre}_loss_consistency'] = losses_consistency[-1]

            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i, loss_iou_pred_i, loss_consistency_i in zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_iou_pred[:-1], losses_consistency[:-1]):
                loss_dict[f'{pre}_d{num_dec_layer}.loss_cls'] = loss_cls_i
                if pre == 'cld':
                    loss_dict[f'{pre}_d{num_dec_layer}.loss_bbox'] = loss_bbox_i
                loss_dict[f'{pre}_d{num_dec_layer}.loss_iou'] = loss_iou_i
                loss_dict[f'{pre}_d{num_dec_layer}.loss_iou_pred'] = loss_iou_pred_i
                loss_dict[f'{pre}_d{num_dec_layer}.loss_consistency'] = loss_consistency_i
                num_dec_layer += 1
                
            # for dn queries
            dn_cls_scores = preds_dicts['dn_cls_scores']
            dn_iou_preds = preds_dicts['dn_iou_preds']
            dn_uncertainty_preds = preds_dicts['dn_uncertainty_preds']

            num_dec_layers = len(dn_cls_scores)
            device = gt_labels_list[0].device

            dn_mask_dict = [preds_dicts['dn_mask_dict'] for _ in range(len(dn_iou_preds))]
            # calculate class and box loss
            dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_iou_pred, dn_losses_consistency = multi_apply(
                self.dn_loss_single, dn_cls_scores, dn_bbox_preds, dn_iou_preds, dn_uncertainty_preds, dn_mask_dict, pre_list)
            if pre == 'opn_':
                dn_losses_bbox = [torch.zeros_like(dn_losses_bbox[0]) for _ in range(len(dn_losses_bbox))]

            # loss from the last decoder layer
            loss_dict['{pre}_dn_loss_cls'] = dn_losses_cls[-1]
            if pre == 'cld':
                loss_dict['{pre}_dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['{pre}_dn_loss_iou'] = dn_losses_iou[-1]
            loss_dict['{pre}_dn_loss_iou_pred'] = dn_losses_iou_pred[-1]
            loss_dict['{pre}_dn_loss_consistency'] = dn_losses_consistency[-1]

            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i, loss_iou_pred_i, loss_consistency_i in zip(dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1], dn_losses_iou_pred[:-1], dn_losses_consistency[:-1]):
                loss_dict[f'{pre}_d{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                if pre == 'cld':
                    loss_dict[f'{pre}_d{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'{pre}_d{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                loss_dict[f'{pre}_d{num_dec_layer}.dn_loss_iou_pred'] = loss_iou_pred_i
                loss_dict[f'{pre}_d{num_dec_layer}.dn_loss_consistency'] = loss_consistency_i
                num_dec_layer += 1

        return loss_dict
    

    def soft_nms(self, boxes, scores, gaussian_sigma=0.3, prune_threshold=1e-3):
        boxes = boxes.clone()
        scores = scores.clone()
        idxs = torch.arange(scores.size()[0]).to(boxes.device)

        idxs_out = []
        scores_out = []

        while scores.numel() > 0:
            top_idx = torch.argmax(scores)
            idxs_out.append(idxs[top_idx].item())
            scores_out.append(scores[top_idx].item())

            top_box = boxes[top_idx]
            ious = bbox_overlaps_3d(top_box.unsqueeze(0), boxes, coordinate='lidar')[0]

            decay = torch.exp(-torch.pow(ious, 2) / gaussian_sigma)

            scores *= decay
            keep = scores > prune_threshold
            keep[top_idx] = False

            # print(keep.device, boxes.device)
            boxes = boxes[keep]
            scores = scores[keep]
            idxs = idxs[keep]

        return torch.tensor(idxs_out).to(boxes.device), torch.tensor(scores_out).to(scores.device)



    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.shape[-1])
            scores = preds['scores']
            labels = preds['labels']
            ious = preds['ious']
            if self.post_processing is not None:
                if self.post_processing['type'] == 'nms' or self.post_processing['type'] == 'soft_nms':
                    nc = self.num_classes
                    nmsbboxes = []
                    nmslabels = []
                    nmsscores = []
                    nmsious = []
                    for j in range(nc):
                        ind = (labels == j)
                        bboxest, labelst, scorest, iousest = bboxes.tensor[ind], labels[ind], scores[ind], ious[ind]
                        if ind.sum() == 0:
                            continue
                        
                        if self.post_processing['type'] == 'nms':
                            nmsind = nms3d(bboxest[:, :7], scorest, self.post_processing['nms_thr'])
                            nmsbboxes.append(bboxest[nmsind])
                            nmsscores.append(scorest[nmsind])
                            nmsious.append(iousest[nmsind])
                            nmslabels.extend([j] * nmsind.shape[0])
                        else:
                            nmsind, scores_soft = self.soft_nms(bboxest[:, :7], scorest, self.post_processing['gaussian_sigma'], self.post_processing['prune_threshold'])
                            nmsbboxes.append(bboxest[nmsind])
                            nmsscores.append(scores_soft)
                            nmsious.append(iousest[nmsind])
                            nmslabels.extend([j] * nmsind.shape[0])
                    if len(nmsbboxes) == 0:
                        nmsbboxes.append(bboxes.tensor.new_zeros((0, bboxes.tensor.shape[-1])))
                        nmsscores.append(bboxes.tensor.new_zeros((0)))
                        nmsious.append(bboxes.tensor.new_zeros((0)))
                    nmsbboxes = torch.cat(nmsbboxes)
                    bboxes = img_metas[i]['box_type_3d'](nmsbboxes, bboxes.tensor.shape[-1])
                    scores = torch.cat(nmsscores)
                    labels = torch.tensor(nmslabels)
                    ious = torch.cat(nmsious)
                elif self.post_processing['type'] == 'box_merging':
                    import projects.mmdet3d_plugin.core.bbox.bbox_merging as bbox_merging
                    class_labels, detection_boxes_3d, detection_scores, nms_indices = bbox_merging.nms_boxes_3d_merge_only(
                        labels.cpu().numpy(), bboxes.tensor.cpu().numpy(), scores.cpu().numpy(),
                        overlapped_fn=bbox_merging.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=0.1, 
                        appr_factor=1e6, top_k=-1,
                        attributes=np.arange(len(labels)))
                    bboxes = img_metas[i]['box_type_3d'](torch.tensor(detection_boxes_3d), bboxes.tensor.shape[-1])
                    scores = torch.tensor(detection_scores)
                    labels = torch.tensor(class_labels)
                    ious = torch.tensor(nms_indices)
                else:
                    raise(self.post_processing['type'] +' not implemented.')

                if 'score_thr' in self.post_processing:
                    if type(self.post_processing['score_thr']) is list:
                        assert len(self.post_processing['score_thr']) == self.num_classes
                        ind = (scores < -1)
                        for j in range(self.num_classes):
                            ind = torch.logical_or(ind, torch.logical_and(labels==j, scores > self.post_processing['score_thr'][j]))
                        bboxes = img_metas[i]['box_type_3d'](bboxes.tensor[ind], bboxes.tensor.shape[-1])
                        scores = torch.tensor(scores[ind])
                        labels = torch.tensor(labels[ind])
                    else:
                        ind = (scores > self.post_processing['score_thr'])
                        bboxes = img_metas[i]['box_type_3d'](bboxes.tensor[ind], bboxes.tensor.shape[-1])
                        scores = torch.tensor(scores[ind])
                        labels = torch.tensor(labels[ind])

                if 'num_thr' in self.post_processing:
                    ind = torch.argsort(-scores)[0:self.post_processing['num_thr']]
                    bboxes = img_metas[i]['box_type_3d'](bboxes.tensor[ind], bboxes.tensor.shape[-1])
                    scores = torch.tensor(scores[ind])
                    labels = torch.tensor(labels[ind])

            ret_list.append([bboxes, scores, labels])
        return ret_list
