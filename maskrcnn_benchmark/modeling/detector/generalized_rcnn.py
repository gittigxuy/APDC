# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        if cfg.VIS_ACC:
            vis_dir = os.path.join(cfg.OUTPUT_DIR, cfg.FILE, "logs")
            self.writer = SummaryWriter(vis_dir)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # -------------------------------------------------------------------
        if cfg.START_ATTENTION:
            assert features[0].size()[0] == 1
            feature = self.roi_heads["box"].feature_extractor.head(features[0])
            cls_score = self.roi_heads["box"].predictor.cls_score
            feature = feature*cls_score.weight[1].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            feature = feature.sum(dim=1).unsqueeze(0)
            feature = F.interpolate(feature, size=(features[0].size()[2], features[0].size()[3]), mode='bilinear', align_corners=True)
            feature = feature * features[0] + features[0]
            features = [feature]
        # -------------------------------------------------------------------
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses, keep = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        output = {"detections": result, "proposals": proposals, "keep": keep}
        return output
