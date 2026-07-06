# RCT-AD Architecture & Equation-to-Code Map

This document maps every equation and algorithm in the manuscript to the code
that implements it, so reviewers and users can trace the method end-to-end.

## Overview

RCT-AD encodes multi-view camera images into a unified BEV representation, then
refines that representation with the **Reliable Context Awareness (RCA)** module
before feeding it to the **Detection & Segmentation Head** and the **Temporal
Trajectory Planner**. All heads are trained jointly on a shared BEV latent.

```
images → backbone+neck → BEV (Eq.1) → RCA (Eq.2–8) → { Det&Seg (Eq.13–14),
                                                        TTP (Eq.9–12) } → L_total (Eq.15)
```

## Equation → Code map

| Eq.  | Description                                   | Location |
|------|-----------------------------------------------|----------|
| (1)  | Multi-view BEV encoding `F_BEV = Proj(Neck(Backbone(I)))` | `models/rct_ad.py::RCTAD.extract_feat` |
| (2)  | Reliability score `R_t = Σ αᵢ·qualityᵢ`       | `rca/…::ReliableContextAwareness.compute_reliability` |
| (3)  | Most-reliable historical retrieval `argmax_m R_m` | `rca/…::_most_reliable_historical` |
| (4)  | Warp-and-blend meta-update, `λ^(k)`           | `rca/…::bounded_meta_update` |
| (5)  | FILO short-term buffer push `M_t`             | `rca/…::_push_short_term` |
| (6)  | Reliability-gated instance update `γ_t = g(R_t)` | `rca/…::reliability_gate`, used in `reliability_fusion` |
| (7)  | Reliability-weighted fusion `F_t^fused`       | `rca/…::reliability_fusion` |
| (8)  | Channel-attention refinement `F_t^final`      | `rca/…::ChannelAttention`, `reliability_fusion` |
| Alg.1| RCA routing (compute→route→repair→maintain→fuse) | `rca/…::forward` |
| (9)  | LSTM temporal sequence encoding `H_t^m`       | `motion/temporal_trajectory_planner.py::encode_temporal` |
| (10) | `Z_t^m = CrossGNN(TempGNN(H,A), M_map)`        | `…::spatial_interaction` |
| (11) | Multi-modal decoding (cls/reg/state)          | `…::decode` |
| (12) | Planning objective `L_plan`                   | `…::TemporalTrajectoryPlanner.loss` |
| (13) | Motion-for-detection fusion `q'_t = Attn(q_t+φ(m_t))` | `det_seg/…::MotionForDetectionFusion` |
| (14) | BEV semantic segmentation `Y_seg`             | `det_seg/…::SemanticBEVSegmentation` |
| (15) | Unified multi-task loss `L_total`             | `models/rct_ad_head.py::RCTADHead.loss` + config loss weights |

## Reliable Context Awareness (RCA), in detail

RCA maintains two stores sharing one reliability signal:

1. **Short-term Reliable Features** — a FILO buffer of `(feature, reliability,
   age)` tuples. Importance decays as `wᵢ = exp(-β·aᵢ)`. When full, the oldest /
   least-reliable entry is evicted while strong entries are refreshed.

2. **Long-term Reliable Memory Bank** — stores reliable per-frame / per-instance
   embeddings. At decode time, all entries are fused by reliability weights
   (Eq. 7) and refined by channel attention (Eq. 8).

**Routing (Algorithm 1).** For each frame with score `R_t`:
- `R_t ≥ τ` → promote to Reliable Memory Bank (reliable path).
- `R_t < τ` → bounded meta-update: repeatedly blend with the most reliable
  historical feature and re-score, for at most `K_r` iterations. If reliability
  is restored, accept the repaired feature; otherwise fall back to the warped
  historical feature. The accepted feature is pushed into the FILO buffer.

The threshold `τ = 0.85`, budget `K_r`, capacity `K`, decay `β`, and the
coefficients `α₁..α₅` are set in the config (see below).

## Configurable hyperparameters

Defined at the top of `projects/configs/RCT-AD_stage2.py`:

```python
use_rca            = True
rca_tau            = 0.85
rca_max_meta_iters = 3
rca_queue_length   = 6
rca_alphas         = (0.25, 0.30, 0.15, 0.20, 0.10)   # IoU, Conf, 1-H, S, P
rca_beta           = 0.2
```

## Ablations reproduced by config toggles

| Ablation (paper table) | How to reproduce |
|------------------------|------------------|
| w/o RCA (Table VI)     | set `use_rca = False` |
| w/o FILO (Table VII)   | set `rca_queue_length = 1` (no short-term history) |
| threshold sweep (Table VIII) | set `rca_tau ∈ {0.65, 0.75, 0.85}` |
| coefficient sensitivity (Table IX) | edit `rca_alphas` (must sum to 1) |
