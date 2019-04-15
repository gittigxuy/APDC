# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_iod, boxlist_iog
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_visible_rate
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
            self,
            proposal_matcher,
            proposal_matcher_with_visible_iog,
            fg_bg_sampler,
            box_coder,
            cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.proposal_matcher_with_visible_iog = proposal_matcher_with_visible_iog
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target, posv_targets):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def match_both_targets_to_proposals(self, proposal, target, posv_targets):
        match_quality_matrix = boxlist_iou(target, proposal)
        match_quality_matrix_iog = boxlist_iog(posv_targets, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        matched_with_visible_idxs = self.proposal_matcher_with_visible_iog(match_quality_matrix_iog)

        index_out_bounds = np.where(matched_idxs.cpu().numpy() == -2)[0]
        matched_both = np.where((matched_idxs.cpu().numpy() > -1) & (matched_with_visible_idxs.cpu().numpy() > -1))[0]
        matched_both = torch.tensor(matched_both, dtype=matched_idxs.dtype, device=matched_idxs.device)
        index_out_bounds = torch.tensor(index_out_bounds, dtype=matched_idxs.dtype, device=matched_idxs.device)
        matched_new = -1 * torch.ones_like(matched_idxs, dtype=matched_idxs.dtype)  # background
        matched_new[matched_both] = matched_idxs[matched_both]  # TODO: label is important
        matched_new[index_out_bounds] = matched_idxs[index_out_bounds]  # out of bounds
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_new.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_new)
        return matched_targets

    def match_posv_targets_to_proposals(self, proposal, target, posv_targets):
        match_quality_matrix = boxlist_iou(posv_targets, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets, posv_targets=None):
        labels = []
        regression_targets = []
        regression_targets_visible_rate = []
        for proposals_per_image, targets_per_image, posv_targets_per_image in zip(proposals, targets, posv_targets):
            if cfg.MODEL.ROI_BOX_HEAD.USE_VISIBLE_BODY:
                matched_targets = self.match_posv_targets_to_proposals(
                    proposals_per_image, targets_per_image, posv_targets_per_image
                )
            else:
                matched_targets = self.match_targets_to_proposals(
                    proposals_per_image, targets_per_image, posv_targets_per_image
                )
                # matched_targets = self.match_both_targets_to_proposals(
                #     proposals_per_image, targets_per_image, posv_targets_per_image
                # )
            visible_rate = boxlist_visible_rate(proposals_per_image, posv_targets_per_image, targets_per_image)
            visible_rate_max, visible_rate_matches = visible_rate.max(dim=1)
            visible_rate_max = visible_rate_max.unsqueeze(1)

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            # and target be labeled ignore
            ignore = targets_per_image.extra_fields['ignore']
            targets_inds = torch.arange(ignore.size()[0]).cuda()
            ignore_inds_set = targets_inds[ignore == 1]
            for index in ignore_inds_set:
                ignore_inds = matched_idxs == index
                labels_per_image[ignore_inds] = -1
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            regression_targets_visible_rate.append(visible_rate_max)
        return labels, regression_targets, regression_targets_visible_rate

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        posv_targets = []
        for target in targets:
            target.extra_fields['posv'].add_field("labels", target.extra_fields['labels'])
            posv_targets.append(target.extra_fields['posv'])
        # this code for pad proposals
        # proposals_scale = []
        # for p in proposals:
        #     proposals_scale.append(p.expand_boxes(cfg.MODEL.RPN.PROPOSALS_PAD))
        # labels, regression_targets, regression_targets_visible_rate = self.prepare_targets(
        #     proposals, proposals_scale, targets, posv_targets)
        # proposals = list(proposals_scale)
        labels, regression_targets, regression_targets_visible_rate = self.prepare_targets(
            proposals, targets, posv_targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, regression_targets_visible_rate_per_image, proposals_per_image in zip(
            labels, regression_targets, regression_targets_visible_rate, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field(
                "regression_targets_visible_rate", regression_targets_visible_rate_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])
        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
            visible_rate_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)
        classification_loss = classification_loss * cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_LOSS_BETA
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)
        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()  # TODO:?? is labels.numel()?

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher_iou = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_LOW_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    matcher_visible_iog = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOG_VISIBLE_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOG_VISIBLE_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher_iou,
        matcher_visible_iog,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
