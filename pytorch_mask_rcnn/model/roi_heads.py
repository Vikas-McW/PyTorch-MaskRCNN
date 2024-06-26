import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch import nn  # type: ignore

from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    return classifier_loss, box_reg_loss


def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    return mask_loss
    

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                fg_iou_thresh, bg_iou_thresh,
                num_samples, positive_fraction,
                reg_weights,
                score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
        
    def has_mask(self):
        return False if self.mask_roi_pool is None else self.mask_predictor is not None


    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)
        
        boxes = []
        labels = []
        scores = []
        for l in range(1, num_classes):
            score, box_delta = pred_score[:, l], box_regression[:, l]

            keep = score >= self.score_thresh
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)
            
            box, score = process_box(box, score, image_shape, self.min_size)
            
            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)
            
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        return results
    
    def forward(self, feature, proposal, image_shape):
        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)
        
        result, losses = {}, {}
        result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)
        if self.has_mask():
        
            mask_proposal = result['boxes']
            
            if mask_proposal.shape[0] == 0:
                result.update(dict(masks=torch.empty((0, 28, 28))))
                return result, losses
                
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)
            
            label = result['labels']
            idx = torch.arange(label.shape[0], device=label.device)
            mask_logit = mask_logit[idx, label]
            mask_prob = mask_logit.sigmoid()
            result.update(dict(masks=mask_prob))
                
        return result, losses
    
    

    
    