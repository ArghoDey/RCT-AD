"""Detection and Segmentation Head.

Implements the perception-refinement stage of RCT-AD described in Section III-D
(Eqs. 13-14) of the manuscript. Instead of handling detection and segmentation
separately, this head merges them within a unified refinement layer that
enriches the shared BEV features with temporal motion and scene semantics. The
integration sharpens object boundaries, stabilizes localization under motion
blur / occlusion, and keeps perception and planning scene-consistent.

The head has two complementary parts:

  (1) Motion-for-Detection Adaptive Fusion (Eq. 13):
        q'_t = Attn( q_t + phi(m_t) )
      A compact transformer block injects temporal motion embeddings m_t into
      the detection queries via motion-guided cross-attention, improving object
      localization in dense / occluded scenes.

  (2) Semantic BEV Segmentation for Guided Supervision (Eq. 14):
        Y_seg = Conv_{1x1}( ReLU( BN( Conv_{3x3}(F_BEV) ) ) )
      A BEV semantic head reshapes BEV tokens into a spatial grid and predicts
      per-cell classes (roads, lanes, drivable areas, pedestrians, ...),
      providing topological priors that guide both detection and planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmdet.models import HEADS

__all__ = ["DetectionSegmentationHead"]


class MotionForDetectionFusion(nn.Module):
    """Motion-for-Detection Adaptive Fusion (Eq. 13).

    Embeds temporal motion information into the detection decoder through
    motion-guided cross-attention.
    """

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        # phi: learned projection aligning temporal motion with spatial features.
        self.phi = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, det_queries, motion_embeds):
        """Return motion-enhanced detection queries q'_t.

        Args:
            det_queries (Tensor): (B, N_q, C) detection queries q_t.
            motion_embeds (Tensor): (B, N_q, C) temporal motion embeddings m_t.
        """
        fused = det_queries + self.phi(motion_embeds)  # q_t + phi(m_t)
        attn_out, _ = self.attn(fused, fused, fused)  # Attn(.)
        return self.norm(det_queries + attn_out)


class SemanticBEVSegmentation(nn.Module):
    """Semantic BEV Segmentation head (Eq. 14).

    Y_seg = Conv_{1x1}( ReLU( BN( Conv_{3x3}(F_BEV) ) ) )
    """

    def __init__(self, embed_dims=256, num_seg_classes=7):
        super().__init__()
        self.conv3x3 = nn.Conv2d(embed_dims, embed_dims, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(embed_dims)
        self.conv1x1 = nn.Conv2d(embed_dims, num_seg_classes, 1)

    def forward(self, bev_feat):
        """Predict per-cell semantic classes from the BEV feature grid.

        Args:
            bev_feat (Tensor): (B, C, H, W) shared BEV feature.

        Returns:
            Tensor: (B, num_seg_classes, H, W) segmentation logits.
        """
        x = self.conv1x1(F.relu(self.bn(self.conv3x3(bev_feat))))
        return x


@HEADS.register_module()
class DetectionSegmentationHead(BaseModule):
    """Unified Detection and Segmentation Head (Section III-D).

    Args:
        embed_dims (int): BEV / query channel dimension.
        num_heads (int): attention heads for motion-guided fusion.
        num_seg_classes (int): number of BEV semantic classes.
        dropout (float): dropout rate.
        seg_loss_weight (float): weight of the segmentation loss.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_seg_classes=7,
        dropout=0.1,
        seg_loss_weight=1.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.seg_loss_weight = seg_loss_weight

        self.motion_for_det = MotionForDetectionFusion(embed_dims, num_heads, dropout)
        self.seg_head = SemanticBEVSegmentation(embed_dims, num_seg_classes)

    def forward(self, bev_feat, det_queries=None, motion_embeds=None):
        """Refine detection queries and predict BEV semantics.

        Args:
            bev_feat (Tensor): (B, C, H, W) fused BEV feature F_t^fused.
            det_queries (Tensor): optional (B, N_q, C) detection queries q_t.
            motion_embeds (Tensor): optional (B, N_q, C) motion embeddings m_t.

        Returns:
            dict with 'det_queries' (motion-enhanced, if inputs given) and
            'seg_logits' (BEV semantic segmentation).
        """
        out = {}
        if det_queries is not None and motion_embeds is not None:
            out["det_queries"] = self.motion_for_det(det_queries, motion_embeds)
        out["seg_logits"] = self.seg_head(bev_feat)
        return out

    def loss(self, seg_logits, gt_seg):
        """Cross-entropy segmentation loss for guided supervision.

        Args:
            seg_logits (Tensor): (B, num_classes, H, W).
            gt_seg (Tensor): (B, H, W) integer class map.
        """
        loss_seg = F.cross_entropy(seg_logits, gt_seg.long())
        return dict(loss_seg=self.seg_loss_weight * loss_seg)
