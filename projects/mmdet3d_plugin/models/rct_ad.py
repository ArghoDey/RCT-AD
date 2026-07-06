"""RCT-AD top-level detector.

RCT-AD (Reliable Context-Aware and Temporal Planning framework for Autonomous
Driving) is a semantically guided, memory-gated end-to-end model that unifies
perception, temporal reasoning and planning within a single shared BEV
representation (see Fig. 2 of the manuscript).

This detector extends the multi-view BEV backbone (ResNet-50/101 + deformable
neck, Eq. 1) with the Reliable Context Awareness (RCA) module, so that the
shared BEV feature fed to every downstream head is reliability-refined before
fusion. The remaining heads (detection, map/segmentation, motion, planning) are
orchestrated by :class:`RCTADHead`.

The class is deliberately a thin wrapper over the proven multi-view encoder:
the novelty of RCT-AD lives in the RCA / Temporal Trajectory Planner /
Detection-and-Segmentation modules, and keeping the detector minimal makes the
contribution boundaries clear and the pipeline easy to ablate.
"""

from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except Exception:
    DAF_VALID = False

__all__ = ["RCTAD"]


@DETECTORS.register_module()
class RCTAD(BaseDetector):
    """Reliable Context-Aware and Temporal Planning detector.

    Args mirror the multi-view BEV detector; ``rca`` optionally builds the
    Reliable Context Awareness module that refines BEV features before they are
    passed to the heads.
    """

    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        rca=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(RCTAD, self).__init__(init_cfg=init_cfg)

        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func

        # Reliable Context Awareness module (optional, for ablation).
        if rca is not None:
            self.rca = build_from_cfg(rca, PLUGIN_LAYERS)
        else:
            self.rca = None

        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        """Encode multi-view images into a unified BEV representation (Eq. 1)."""
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    def _apply_rca(self, feature_maps, metas):
        """Refine the shared BEV feature through Reliable Context Awareness.

        RCA operates on a dense BEV grid. When the backbone emits multi-scale
        feature maps, the finest level is treated as the shared BEV feature and
        refined in place; sequence boundaries reset the RCA memory.
        """
        if self.rca is None:
            return feature_maps
        is_new = False
        if metas is not None and "img_metas" in metas:
            # A frame is a new sequence when it has no previous ego transform.
            first = metas["img_metas"][0]
            is_new = bool(first.get("prev_exists", 0) == 0)
        # feature_maps[0] is used as the dense BEV proxy when available.
        bev = feature_maps[0] if isinstance(feature_maps, (list, tuple)) else feature_maps
        if bev.dim() == 5:  # (B, num_cams, C, H, W) -> pool cams for RCA scoring
            bev_grid = bev.mean(dim=1)
        else:
            bev_grid = bev
        refined, _ = self.rca(bev_grid, indicators=None, is_new_sequence=is_new)
        # Broadcast the refined grid back if we pooled the camera dimension.
        if bev.dim() == 5:
            feature_maps = list(feature_maps)
            feature_maps[0] = bev + (refined.unsqueeze(1) - bev_grid.unsqueeze(1))
        else:
            feature_maps = list(feature_maps) if isinstance(feature_maps, (list, tuple)) else refined
            if isinstance(feature_maps, list):
                feature_maps[0] = refined
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        feature_maps = self._apply_rca(feature_maps, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)
        feature_maps = self._apply_rca(feature_maps, data)
        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
