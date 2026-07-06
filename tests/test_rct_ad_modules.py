"""Unit tests for the three RCT-AD contributions.

These tests exercise the forward/loss logic of the novel modules in isolation,
without requiring the full mmdet/mmcv/CUDA stack. They verify output shapes,
routing behaviour, gradient flow, and memory bookkeeping.

Run with:
    python -m pytest tests/ -v
"""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "projects", "mmdet3d_plugin", "models",
)


# --------------------------------------------------------------------------- #
# Minimal registry stubs so the modules import without mmcv/mmdet installed.
# --------------------------------------------------------------------------- #
def _install_stubs():
    import types

    class _Reg:
        def register_module(self, *a, **k):
            if a and callable(a[0]):
                return a[0]
            def deco(cls):
                return cls
            return deco

    def _mk(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    import torch.nn as nn

    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()
            self.init_cfg = init_cfg

    _mk("mmcv")
    _mk("mmcv.runner", BaseModule=BaseModule)
    _mk("mmcv.cnn")
    _mk("mmcv.cnn.bricks")
    _mk(
        "mmcv.cnn.bricks.registry",
        PLUGIN_LAYERS=_Reg(), ATTENTION=_Reg(), POSITIONAL_ENCODING=_Reg(),
        FEEDFORWARD_NETWORK=_Reg(), NORM_LAYERS=_Reg(),
    )
    _mk("mmdet")
    _mk("mmdet.models", HEADS=_Reg(), DETECTORS=_Reg())


_install_stubs()


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rca_mod = _load("rca_m", "rca/reliable_context_awareness.py")
ttp_mod = _load("ttp_m", "motion/temporal_trajectory_planner.py")
ds_mod = _load("ds_m", "det_seg/detection_segmentation_head.py")


# --------------------------------------------------------------------------- #
# Reliable Context Awareness
# --------------------------------------------------------------------------- #
class TestRCA:
    def test_forward_shape_and_memory(self):
        rca = rca_mod.ReliableContextAwareness(embed_dims=32, queue_length=4)
        rca.eval()
        bev = torch.randn(2, 32, 16, 16)
        out, R = rca(bev, is_new_sequence=True)
        assert out.shape == bev.shape
        assert 0.0 <= float(R.mean()) <= 1.0
        assert len(rca.memory_bank) >= 1

    def test_capacity_is_respected(self):
        rca = rca_mod.ReliableContextAwareness(embed_dims=16, queue_length=3)
        rca.eval()
        for t in range(10):
            rca(torch.randn(1, 16, 8, 8), is_new_sequence=(t == 0))
        assert len(rca.memory_bank) <= 3
        assert len(rca.short_term_buffer) <= 3

    def test_degraded_frame_triggers_repair(self):
        rca = rca_mod.ReliableContextAwareness(embed_dims=16, tau=0.85, queue_length=4)
        rca.eval()
        # seed memory with a reliable frame
        rca(torch.randn(1, 16, 8, 8),
            indicators=dict(iou=0.9, conf=0.9, entropy=0.1, stability=0.9, clarity=0.9),
            is_new_sequence=True)
        n_before = len(rca.short_term_buffer)
        # a degraded frame should be routed into the FILO buffer
        rca(torch.randn(1, 16, 8, 8),
            indicators=dict(iou=0.1, conf=0.1, entropy=0.9, stability=0.1, clarity=0.1))
        assert len(rca.short_term_buffer) == n_before + 1

    def test_alphas_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            rca_mod.ReliableContextAwareness(embed_dims=16, alphas=(0.5, 0.5, 0.5, 0.5, 0.5))

    def test_gradients_flow(self):
        rca = rca_mod.ReliableContextAwareness(embed_dims=16, queue_length=4)
        out, R = rca(torch.randn(1, 16, 8, 8), is_new_sequence=True)
        (out.mean() + R.mean()).backward()
        trained = [p for p in rca.parameters() if p.requires_grad]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained)


# --------------------------------------------------------------------------- #
# Temporal Trajectory Planner
# --------------------------------------------------------------------------- #
class TestTTP:
    def _make(self):
        return ttp_mod.TemporalTrajectoryPlanner(
            embed_dims=32, ego_fut_ts=6, ego_fut_mode=6, num_maneuvers=3
        )

    def test_forward_shapes(self):
        ttp = self._make()
        pred = ttp(torch.randn(2, 5, 4, 32), torch.randn(2, 8, 32))
        assert pred["cls"].shape == (2, 5, 6, 3)
        assert pred["traj"].shape == (2, 5, 6, 6, 2)
        assert pred["state"].shape == (2, 5, 6, 2)

    def test_single_agent_sequence(self):
        ttp = self._make()
        pred = ttp(torch.randn(2, 4, 32))  # (B, T, C)
        assert pred["traj"].shape == (2, 1, 6, 6, 2)

    def test_loss_and_backward(self):
        ttp = self._make()
        pred = ttp(torch.randn(2, 5, 4, 32), torch.randn(2, 8, 32))
        gt = dict(
            waypoints=torch.randn(2, 5, 6, 2),
            state=torch.randn(2, 5, 2),
            maneuver=torch.zeros(2, 5, dtype=torch.long),
        )
        losses = ttp.loss(pred, gt)
        assert "loss_plan" in losses
        losses["loss_plan"].backward()


# --------------------------------------------------------------------------- #
# Detection & Segmentation Head
# --------------------------------------------------------------------------- #
class TestDetSeg:
    def _make(self):
        return ds_mod.DetectionSegmentationHead(embed_dims=32, num_seg_classes=7)

    def test_seg_only(self):
        ds = self._make()
        out = ds(torch.randn(2, 32, 24, 24))
        assert out["seg_logits"].shape == (2, 7, 24, 24)
        assert "det_queries" not in out

    def test_motion_fusion(self):
        ds = self._make()
        out = ds(torch.randn(2, 32, 24, 24),
                 det_queries=torch.randn(2, 50, 32),
                 motion_embeds=torch.randn(2, 50, 32))
        assert out["det_queries"].shape == (2, 50, 32)

    def test_seg_loss_and_backward(self):
        ds = self._make()
        out = ds(torch.randn(2, 32, 24, 24))
        loss = ds.loss(out["seg_logits"], torch.randint(0, 7, (2, 24, 24)))
        assert "loss_seg" in loss
        loss["loss_seg"].backward()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
