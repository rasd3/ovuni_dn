import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from mmdet.models.utils.transformer import inverse_sigmoid


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 alpha=0.5,
                 num_classes=10):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.alpha = alpha

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, all_iou_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        #max_num = cls_scores.numel()

        cls_scores = cls_scores.sigmoid()
        ious = all_iou_preds.sigmoid()

        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        all_iou_preds = all_iou_preds.sigmoid()
        final_ious = all_iou_preds[bbox_index]

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            # self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            self.post_center_range = scores.new_tensor(self.post_center_range)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            ious = final_ious[mask]
            
            predictions_dict = {
                'bboxes': boxes3d,
                #'scores': scores, 
                'scores': scores ** self.alpha * ious.reshape(-1) ** (1-self.alpha),
                'labels': labels,
                'ious': ious.reshape(-1),
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # all_cls_scores = preds_dicts['all_cls_scores'][-1]
        # all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        # all_iou_preds = preds_dicts['all_iou_preds'][-1]

        all_cls_scores = torch.mean(preds_dicts['all_cls_scores'][1:], 0)
        all_bbox_preds = torch.mean(preds_dicts['all_bbox_preds'][1:], 0)
        all_iou_preds = torch.mean(preds_dicts['all_iou_preds'][1:], 0)

        #all_centerness_preds = torch.mean(preds_dicts['all_centerness_preds'][1:], 0)
        # all_cls_scores = torch.mean(preds_dicts['all_cls_scores'], 0)
        # all_bbox_preds = torch.mean(preds_dicts['all_bbox_preds'], 0)
        # all_cls_scores = 0. * preds_dicts['all_cls_scores'][0] + 0.4 * preds_dicts['all_cls_scores'][1] + 0.6 * preds_dicts['all_cls_scores'][2]
        # all_bbox_preds = 0. * preds_dicts['all_bbox_preds'][0] + 0.4 * preds_dicts['all_bbox_preds'][1] + 0.6 * preds_dicts['all_bbox_preds'][2]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i], all_iou_preds[i]))
            #predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i], all_iou_preds[i], all_centerness_preds[i]))
        return predictions_list

    def decode_agn(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # all_cls_scores = preds_dicts['all_cls_scores'][-1]
        # all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        # all_iou_preds = preds_dicts['all_iou_preds'][-1]

        all_cls_agn_scores = torch.mean(preds_dicts['all_cls_agn_scores'][1:], 0)
        all_bbox_preds = torch.mean(preds_dicts['all_bbox_preds'][1:], 0)
        all_iou_preds = torch.mean(preds_dicts['all_iou_preds'][1:], 0)

        #all_centerness_preds = torch.mean(preds_dicts['all_centerness_preds'][1:], 0)
        # all_cls_scores = torch.mean(preds_dicts['all_cls_scores'], 0)
        # all_bbox_preds = torch.mean(preds_dicts['all_bbox_preds'], 0)
        # all_cls_scores = 0. * preds_dicts['all_cls_scores'][0] + 0.4 * preds_dicts['all_cls_scores'][1] + 0.6 * preds_dicts['all_cls_scores'][2]
        # all_bbox_preds = 0. * preds_dicts['all_bbox_preds'][0] + 0.4 * preds_dicts['all_bbox_preds'][1] + 0.6 * preds_dicts['all_bbox_preds'][2]
        
        batch_size = all_cls_agn_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_agn_scores[i], all_bbox_preds[i], all_iou_preds[i]))
            #predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i], all_iou_preds[i], all_centerness_preds[i]))
        return predictions_list
