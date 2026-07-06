

> ## A Reliable Context-Aware and Temporal Planning Framework for Autonomous Driving  ##
> 
> Argho Dey, Yunfei Yin, Swachha Ray, Md Minhazul Islam, Zheng Yuan, Sijing Xiong, Hongyu Liu, Zhiqiu Huang
> 
> *Submitted to IEEE Transactions on Intelligent Transportation Systems (T-ITS)*

RCT-AD is a semantically guided, memory-gated **end-to-end autonomous driving**
framework that unifies perception, temporal reasoning, and motion planning
within a single shared Bird's-Eye-View (BEV) representation. Unlike prior
end-to-end systems that aggregate temporal BEV features without assessing the
reliability of the underlying observations, RCT-AD explicitly models **feature
quality** and **temporal consistency** so that corrupted inputs (from occlusion,
motion blur, illumination change, or sensor noise) do not destabilize the scene
representation or the planned trajectory.

<div align="center">
  <img src="assets/pipeline.png" width="90%"/>
</div>

## Highlights

- **Reliable Context Awareness (RCA)** — a reliability-aware feature-routing
  module that scores per-frame BEV quality, promotes trustworthy features to a
  long-term **Reliable Memory Bank**, and repairs degraded frames through a
  bounded meta-update over a quality-gated **First-In-Last-Out (FILO)** buffer.
- **Temporal Trajectory Planner (TTP)** — an LSTM + TempGNN/CrossGNN planner
  that captures long-term temporal dependencies and multi-agent interactions to
  produce smoother, safety-aware, multi-modal trajectories.
- **Detection and Segmentation Head** — a unified refinement stage that injects
  motion cues (motion-guided cross-attention) and semantic supervision (BEV
  semantic segmentation) into the shared BEV space.
- **Unified Multi-Task Joint Optimization** — detection, segmentation,
  prediction, planning, and mapping are trained end-to-end on one BEV latent.

<div align="center">
  <img src="assets/Supplementary Video Output.gif" width="90%"/>
</div>

## Results on nuScenes

| Model      | Backbone | mAP  | NDS  | mIoU | Avg L2 (m) ↓ | Avg Col. (%) ↓ |
|------------|----------|------|------|------|--------------|----------------|
| RCT-AD-T   | R50      | 44.2 | 55.6 | 49.8 | 0.59         | 0.07           |
| RCT-AD-L   | R101     | 52.9 | 61.5 | 52.3 | 0.54         | 0.06           |

RCT-AD runs at **7.2 FPS** (RTX A6000), faster than UniAD while maintaining
competitive memory usage. See the paper for full comparison tables
(planning, prediction, detection, segmentation, tracking, and ablations).

## Architecture

```
Multi-view images
      │
      ▼
ResNet-50/101 + Deformable Neck ──► unified BEV feature  F_BEV        (Eq. 1)
      │
      ▼
Reliable Context Awareness (RCA)                                      (Eqs. 2–8)
  • reliability scoring  R_t = Σ αᵢ · qualityᵢ
  • routing: reliable → Reliable Memory Bank; unreliable → repair loop
  • bounded meta-update + FILO short-term buffer
  • reliability-weighted fusion + channel attention → F_t^final
      │
      ├──► Detection & Segmentation Head                             (Eqs. 13–14)
      │       • Motion-for-Detection adaptive fusion
      │       • Semantic BEV segmentation
      │
      └──► Temporal Trajectory Planner                              (Eqs. 9–12)
              • LSTM temporal encoding
              • TempGNN → CrossGNN spatial interaction
              • multi-modal trajectory decoding (cls / reg / state)
      │
      ▼
Unified Multi-Task Joint Optimization                                (Eq. 15)
```

The three RCT-AD contributions live in dedicated, self-contained modules:

| Contribution                    | File |
|---------------------------------|------|
| Reliable Context Awareness      | `projects/mmdet3d_plugin/models/rca/reliable_context_awareness.py` |
| Temporal Trajectory Planner     | `projects/mmdet3d_plugin/models/motion/temporal_trajectory_planner.py` |
| Detection & Segmentation Head   | `projects/mmdet3d_plugin/models/det_seg/detection_segmentation_head.py` |
| Top-level detector / multi-head | `projects/mmdet3d_plugin/models/rct_ad.py`, `rct_ad_head.py` |

## Getting Started

### Environment & data

RCT-AD builds on the `mmdet` / `mmcv` BEV perception stack. Please follow the
environment and nuScenes data-preparation instructions from
[SparseDrive](https://github.com/swc-17/SparseDrive), then install the
additional requirements:

```bash
pip install -r requirements.txt
# build the deformable aggregation op
cd projects/mmdet3d_plugin/ops && python setup.py build install && cd -
```

Expected data layout:

```
data/
  nuscenes/          # raw nuScenes
  infos/             # generated annotation pkls
  kmeans/            # kmeans_motion_*.npy, kmeans_plan_*.npy anchors
ckpt/
  resnet50-19c8e357.pth
```

### Train

```bash
sh scripts/train.sh          # stage 1 (perception pre-train) → stage 2 (E2E)
```

### Test

```bash
sh scripts/test.sh           # set your checkpoint path inside the script
```

### Configuration

Key RCT-AD hyperparameters are exposed at the top of each config
(`projects/configs/RCT-AD_stage2.py`):

```python
use_rca            = True
rca_tau            = 0.85           # reliability threshold τ (Table VIII)
rca_max_meta_iters = 3              # bounded meta-update budget K_r
rca_queue_length   = 6             # FILO / memory capacity K
rca_alphas         = (0.25, 0.30, 0.15, 0.20, 0.10)  # α₁..α₅ (Table IX)
```

Set `use_rca = False` to reproduce the no-RCA ablation row (Table VI/VII).

## Unit tests

Lightweight, dependency-free tests for the three novel modules:

```bash
python -m pytest tests/ -v
```

## Acknowledgements

All rights are reserved by Argho Dey. RCT-AD is built on top of excellent prior work. We thank the authors of:

- [SparseDrive](https://github.com/swc-17/SparseDrive)
- [BridgeAD](https://github.com/zbozhou/BridgeAD) (our baseline)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [NeuroNCAP](https://github.com/atonderski/neuro-ncap)

## Citation

```bibtex
@article{dey2026rctad,
  title   = {A Reliable Context-Aware and Temporal Planning Framework for Autonomous Driving},
  author  = {Dey, Argho and Yin, Yunfei and Ray, Swachha and Islam, Md Minhazul and
             Yuan, Zheng and Xiong, Sijing and Liu, Hongyu and Huang, Zhiqiu},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
}
```

## License

Released under the terms in [LICENSE](LICENSE).

