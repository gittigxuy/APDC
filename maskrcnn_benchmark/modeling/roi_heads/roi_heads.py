# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .box_separate_head.box_head import build_roi_separate_head
from .box_head_visible_rate.box_head import build_roi_box_visible_rate_head
from .average_box_head.box_head import build_roi_box_average_box_head
from .mask_head.mask_head import build_roi_mask_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box, keep = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        return x, detections, losses, keep


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        if cfg.MODEL.ROI_BOX_HEAD.METHOD == "roi_head_with_visible_rate":
            roi_heads.append(("box", build_roi_box_visible_rate_head(cfg)))
        elif cfg.MODEL.ROI_BOX_HEAD.METHOD == "roi_separate_head":
            roi_heads.append(("box", build_roi_separate_head(cfg)))
        elif cfg.MODEL.ROI_BOX_HEAD.METHOD == "roi_head":
            roi_heads.append(("box", build_roi_box_head(cfg)))
        elif cfg.MODEL.ROI_BOX_HEAD.METHOD == "average_box_head":
            roi_heads.append(("box", build_roi_box_average_box_head(cfg)))
        else:
            raise NotImplementedError("Unsupported {} method.".format(cfg.MODEL.ROI_BOX_HEAD.METHOD))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
