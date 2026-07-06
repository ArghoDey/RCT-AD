"""Reliable Context Awareness (RCA) module.

This file implements the Reliable Context Awareness mechanism described in
Section III-B and Algorithm 1 of:

    "A Reliable Context-Aware and Temporal Planning Framework for Autonomous
     Driving (RCT-AD)".

The module improves the temporal stability of the shared Bird's-Eye-View (BEV)
representation. In real driving, camera observations are frequently corrupted
by occlusion, motion blur, illumination change and sensor noise. When such
degraded observations are aggregated indiscriminately over time, the BEV
representation is destabilized and downstream perception / planning degrade.

RCA is composed of two coordinated sub-modules that share a single reliability
signal:

  (1) Reliable Features (short-term): scores every incoming frame with a
      composite reliability score R_t (Eq. 2). Reliable frames (R_t >= tau) are
      promoted to the long-term Reliable Memory Bank. Unreliable frames
      (R_t < tau) are passed through a bounded meta-update repair loop
      (Eqs. 3-5) that blends the corrupted current feature with the most
      reliable historical feature until reliability is restored, or the
      warped historical feature is trusted as a fallback. A FILO short-term
      buffer stores (feature, reliability, age) tuples with exponential
      age-decay.

  (2) Reliability Fusion (long-term): the Reliable Memory Bank maintains
      per-instance embeddings updated by a learned reliability gate (Eq. 6),
      and produces the final fused BEV feature through reliability-weighted
      fusion and channel attention (Eqs. 7-8).

The module is intentionally self-contained so that it can be dropped into the
RCT-AD BEV pipeline and toggled on/off for ablation studies (see Table VI and
Table VII of the manuscript).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["ReliableContextAwareness"]


class ChannelAttention(nn.Module):
    """Lightweight ECA-style channel attention used to fuse the reliability-
    weighted historical feature with the current-frame feature (Eq. 8).

    Reference: Wang et al., "ECA-Net: Efficient Channel Attention" (CVPR 2020),
    cited as [28] in the manuscript.
    """

    def __init__(self, embed_dims, reduction=4):
        super().__init__()
        hidden = max(embed_dims // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dims, bias=True),
        )

    def forward(self, fused, current):
        # fused, current: (B, C, H, W) or (B, N, C)
        if fused.dim() == 4:
            gap = F.adaptive_avg_pool2d(fused, 1).flatten(1)  # (B, C)
            attn = torch.sigmoid(self.mlp(gap))[..., None, None]
            return fused * attn + current * (1 - attn)
        else:
            gap = fused.mean(dim=1)  # (B, C)
            attn = torch.sigmoid(self.mlp(gap)).unsqueeze(1)
            return fused * attn + current * (1 - attn)


@PLUGIN_LAYERS.register_module()
class ReliableContextAwareness(BaseModule):
    """Reliable Context Awareness (RCA) mechanism.

    Args:
        embed_dims (int): channel dimension of the BEV feature map.
        tau (float): reliability threshold tau (default 0.85, Algorithm 1).
        max_meta_iters (int): K_r, maximum bounded meta-update iterations.
        queue_length (int): capacity K of the FILO short-term buffer and the
            number of fused BEV features kept as historical context.
        alphas (tuple): reliability-score coefficients
            (alpha1..alpha5) for (IoU, Conf, 1-H, S, P). Must sum to 1.
            Default (0.25, 0.30, 0.15, 0.20, 0.10) matches Section IV-D.
        beta (float): exponential age-decay rate for FILO buffer entries.
        reduction (int): channel-attention reduction ratio.
    """

    def __init__(
        self,
        embed_dims=256,
        tau=0.85,
        max_meta_iters=3,
        queue_length=6,
        alphas=(0.25, 0.30, 0.15, 0.20, 0.10),
        beta=0.2,
        reduction=4,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        assert abs(sum(alphas) - 1.0) < 1e-6, "reliability weights must sum to 1"
        self.embed_dims = embed_dims
        self.tau = tau
        self.max_meta_iters = max_meta_iters
        self.queue_length = queue_length
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.beta = beta

        # Learned reliability-gating function g(.) of Eq. (6): a sigmoid-
        # activated single-layer MLP that maps per-frame reliability to a
        # blending factor in [0, 1]. Trained jointly with detection/planning.
        self.reliability_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

        # Channel attention for the final refinement of Eq. (8).
        self.channel_attention = ChannelAttention(embed_dims, reduction)

        # A small predictor used to estimate segmentation entropy / clarity
        # style cues directly from the BEV feature when explicit quality
        # indicators are not supplied by the caller. This keeps RCA usable in a
        # pure-BEV setting while still allowing external indicators.
        self.quality_probe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(embed_dims, 5),
            nn.Sigmoid(),
        )

        self.reset()

    # ------------------------------------------------------------------ #
    # Buffer / memory management
    # ------------------------------------------------------------------ #
    def reset(self):
        """Clear the FILO short-term buffer and the Reliable Memory Bank.

        Called at the start of every driving sequence.
        """
        # FILO short-term buffer: list of dicts {feat, R, age}
        self.short_term_buffer = []
        # Long-term Reliable Memory Bank: list of dicts {feat, R}
        self.memory_bank = []

    def _push_short_term(self, feat, reliability):
        """Push a repaired/accepted feature into the FILO short-term buffer.

        Implements the Push(.) operation of Eq. (5) with exponential age-decay
        eviction (Section III-B, "aging-refresh policy").
        """
        # Age all existing entries.
        for entry in self.short_term_buffer:
            entry["age"] += 1
            entry["weight"] = float(torch.exp(torch.tensor(-self.beta * entry["age"])))
        self.short_term_buffer.append(
            dict(feat=feat.detach(), R=float(reliability), age=0, weight=1.0)
        )
        # FILO eviction: when full, drop oldest / least-reliable entry.
        if len(self.short_term_buffer) > self.queue_length:
            # least reliable OR oldest; combine into a promotion score
            scores = [e["R"] * e["weight"] for e in self.short_term_buffer]
            evict_idx = int(min(range(len(scores)), key=lambda i: scores[i]))
            self.short_term_buffer.pop(evict_idx)

    def _push_memory(self, feat, reliability):
        """Promote a reliable frame to the long-term Reliable Memory Bank."""
        self.memory_bank.append(dict(feat=feat.detach(), R=float(reliability)))
        if len(self.memory_bank) > self.queue_length:
            # evict least-reliable entry to preserve strong context
            evict_idx = int(
                min(range(len(self.memory_bank)), key=lambda i: self.memory_bank[i]["R"])
            )
            self.memory_bank.pop(evict_idx)

    def _most_reliable_historical(self):
        """Return arg max_{m in B} R_m (Eq. 3) from the Reliable Memory Bank."""
        pool = self.memory_bank if len(self.memory_bank) > 0 else self.short_term_buffer
        if len(pool) == 0:
            return None, 0.0
        best = max(pool, key=lambda e: e["R"])
        return best["feat"], best["R"]

    # ------------------------------------------------------------------ #
    # Reliability scoring
    # ------------------------------------------------------------------ #
    def compute_reliability(self, feat, indicators=None):
        """Compute the composite reliability score R_t (Eq. 2).

        R_t = a1*IoU + a2*Conf + a3*(1-H) + a4*S + a5*P

        Args:
            feat (Tensor): current BEV feature (B, C, H, W).
            indicators (dict, optional): externally computed quality cues with
                keys {'iou', 'conf', 'entropy', 'stability', 'clarity'}, each a
                scalar or (B,) tensor normalized to [0, 1]. When not provided,
                the cues are estimated from the feature via ``quality_probe``.

        Returns:
            Tensor: reliability score(s) in [0, 1], shape (B,).
        """
        if indicators is None:
            probe = self.quality_probe(feat)  # (B, 5) in [0,1]
            iou, conf, ent, stab, clar = probe.unbind(dim=-1)
            one_minus_h = 1.0 - ent
        else:
            def _get(k, default):
                v = indicators.get(k, default)
                if not torch.is_tensor(v):
                    v = feat.new_tensor(v)
                return v.reshape(-1)
            iou = _get("iou", 0.5)
            conf = _get("conf", 0.5)
            one_minus_h = 1.0 - _get("entropy", 0.5)
            stab = _get("stability", 0.5)
            clar = _get("clarity", 0.5)

        a = self.alphas
        R = a[0] * iou + a[1] * conf + a[2] * one_minus_h + a[3] * stab + a[4] * clar
        return R.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Bounded meta-update (Eqs. 3-5, Algorithm 1 Step 3)
    # ------------------------------------------------------------------ #
    def bounded_meta_update(self, feat, R0, indicators=None):
        """Repair a degraded feature via the bounded meta-update loop.

        Iteratively blends the corrupted current feature with the most reliable
        historical feature (Eq. 4) and re-scores it (Eq. 2). Terminates when
        R_t^(k) >= tau or after K_r iterations, in which case the warped
        historical feature is trusted as fallback.

        Returns:
            (Tensor, float): the accepted feature f~_t and its reliability.
        """
        f_k = feat
        R_k = float(R0)
        m_star, R_m = self._most_reliable_historical()
        if m_star is None:
            # No historical context yet; accept the current feature as-is.
            return f_k, R_k

        for _ in range(self.max_meta_iters):
            lam = R_k / (R_k + R_m + 1e-6)  # lambda^(k), Eq. (4)
            # Warp(f_{m*}, delta_pose) is approximated as identity here because
            # ego-pose alignment is handled upstream by the temporal queue; if
            # a warp function is provided it can be plugged in at this point.
            f_k = lam * f_k + (1.0 - lam) * m_star
            R_k = float(self.compute_reliability(f_k, indicators).mean().detach())
            if R_k >= self.tau:
                return f_k, R_k  # reliability restored

        # Historical fallback: trust consistent history over corrupted input.
        return m_star, R_m

    # ------------------------------------------------------------------ #
    # Long-term reliability fusion (Eqs. 6-8, Algorithm 1 Step 5)
    # ------------------------------------------------------------------ #
    def reliability_fusion(self, current_feat):
        """Aggregate stored reliable features and refine with channel attention.

        Implements reliability-weighted fusion (Eq. 7) followed by the
        channel-attention refinement (Eq. 8).
        """
        pool = self.memory_bank if len(self.memory_bank) > 0 else self.short_term_buffer
        if len(pool) == 0:
            return current_feat

        feats = torch.stack([e["feat"] for e in pool], dim=0)  # (M, B, C, H, W)
        weights = current_feat.new_tensor([e["R"] for e in pool])  # (M,)
        weights = weights / (weights.sum() + 1e-6)
        # Eq. (6): learned reliability gate softly scales each contribution.
        gates = self.reliability_gate(weights.unsqueeze(-1)).squeeze(-1)  # (M,)
        weights = weights * gates
        weights = weights / (weights.sum() + 1e-6)

        fused = (feats * weights[:, None, None, None, None]).sum(dim=0)  # (B,C,H,W)
        final = self.channel_attention(fused, current_feat)  # Eq. (8)
        return final

    # ------------------------------------------------------------------ #
    # Forward: full RCA routing (Algorithm 1)
    # ------------------------------------------------------------------ #
    def forward(self, bev_feat, indicators=None, is_new_sequence=False):
        """Route a BEV feature frame through the RCA mechanism.

        Args:
            bev_feat (Tensor): current-frame BEV feature (B, C, H, W).
            indicators (dict, optional): external reliability cues (see
                ``compute_reliability``).
            is_new_sequence (bool): reset memory at sequence boundaries.

        Returns:
            (Tensor, Tensor): the refined fused BEV feature F_t^final and the
            per-frame reliability score R_t.
        """
        if is_new_sequence:
            self.reset()

        # Step 1: reliability computation (Eq. 2). R_t keeps its graph so the
        # learned quality probe / reliability gate receive gradients; the
        # scalar used only for routing decisions is detached.
        R_t = self.compute_reliability(bev_feat, indicators)
        R_scalar = float(R_t.mean().detach())

        # Step 2: routing.
        if R_scalar >= self.tau:
            # Reliable frame -> promote to Reliable Memory Bank.
            self._push_memory(bev_feat, R_scalar)
            accepted = bev_feat
        else:
            # Step 3: bounded meta-update repair loop.
            accepted, R_acc = self.bounded_meta_update(bev_feat, R_scalar, indicators)
            # Step 4: memory maintenance via FILO short-term buffer.
            self._push_short_term(accepted, R_acc)
            self._push_memory(accepted, R_acc)

        # Step 5: aggregate and refine (Eqs. 7-8).
        final_feat = self.reliability_fusion(accepted)
        return final_feat, R_t
