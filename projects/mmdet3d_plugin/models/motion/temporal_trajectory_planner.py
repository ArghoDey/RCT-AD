"""Temporal Trajectory Planner (TTP).

Implements the core reasoning module of RCT-AD described in Section III-C
(Eqs. 9-12) of the manuscript. Unlike the static, query-alignment planners of
SparseDrive / BridgeAD / GenAD, the Temporal Trajectory Planner preserves a
temporal memory across frames with an LSTM recurrent encoder and reasons over
multi-agent interactions through a TempGNN / CrossGNN dual-attention structure.

Pipeline:
  1. Temporal Sequence Encoding (Eq. 9): a single-layer LSTM propagates motion
     semantics (acceleration, turning, lane merging) across BEV embeddings.
  2. Spatial Interaction Reasoning (Eq. 10):
        Z_t^m = CrossGNN( TempGNN(H_t^m, A), M_map )
     where TempGNN models inter-agent temporal dynamics (yielding, overtaking)
     and CrossGNN integrates topological priors (lane geometry, drivable area).
  3. Multi-modal Trajectory Decoding (Eq. 11): three lightweight MLP heads for
     maneuver classification, waypoint regression and motion-state estimation.
  4. Planning Objective (Eq. 12): a multi-task loss combining the three heads.

This module is written as a self-contained ``nn.Module`` so it can be used
either as a drop-in temporal decoder inside the RCT-AD planning head or
evaluated in isolation. Graph attention uses standard multi-head attention as a
permutation-equivariant interaction operator over the agent/map token sets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["TemporalTrajectoryPlanner"]


class GraphAttentionLayer(nn.Module):
    """A single graph-attention interaction layer.

    Realizes the message passing used by both TempGNN (inter-agent temporal
    dynamics) and CrossGNN (agent-to-map topological priors). Implemented with
    multi-head attention followed by a residual FFN.
    """

    def __init__(self, embed_dims, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 2, embed_dims),
        )

    def forward(self, query, key=None, value=None, key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        attn_out, _ = self.attn(
            query, key, value, key_padding_mask=key_padding_mask
        )
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


@PLUGIN_LAYERS.register_module()
class TemporalTrajectoryPlanner(BaseModule):
    """Temporal Trajectory Planner (Section III-C).

    Args:
        embed_dims (int): BEV / hidden dimension C (default 256, matching the
            BEV embedding dimension per Eq. 9).
        num_heads (int): number of graph-attention heads.
        ego_fut_ts (int): number of future planning steps T_plan for the ego.
        ego_fut_mode (int): number of trajectory modes M (multi-modal decoding).
        num_maneuvers (int): number of maneuver classes for Psi_cls.
        motion_state_dim (int): dimension of the motion-state prediction.
        dropout (float): dropout rate.
        loss_weights (dict): {'cls', 'reg', 'stat'} weights for Eq. (12).
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        ego_fut_ts=6,
        ego_fut_mode=6,
        num_maneuvers=3,
        motion_state_dim=2,
        dropout=0.1,
        loss_weights=dict(cls=0.5, reg=1.0, stat=1.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_maneuvers = num_maneuvers
        self.motion_state_dim = motion_state_dim
        self.loss_weights = loss_weights

        # --- Temporal Sequence Encoding (Eq. 9) ---
        # Single-layer LSTM with hidden dim = C, preserving motion continuity.
        self.lstm = nn.LSTM(
            input_size=embed_dims,
            hidden_size=embed_dims,
            num_layers=1,
            batch_first=True,
        )

        # --- Spatial Interaction Reasoning (Eq. 10) ---
        self.temp_gnn = GraphAttentionLayer(embed_dims, num_heads, dropout)
        self.cross_gnn = GraphAttentionLayer(embed_dims, num_heads, dropout)

        # --- Multi-modal Trajectory Decoding (Eq. 11) ---
        # Mode embeddings expand a per-agent feature into M motion modes.
        self.mode_embed = nn.Parameter(torch.randn(ego_fut_mode, embed_dims))
        # Psi_cls: maneuver classification head.
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_maneuvers),
        )
        # Psi_reg: waypoint regression head (T_plan * 2 coordinates per mode).
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        # Psi_stat: motion-state estimation head.
        self.stat_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, motion_state_dim),
        )

    # ------------------------------------------------------------------ #
    def encode_temporal(self, seq):
        """Temporal Sequence Encoding, Eq. (9).

        Args:
            seq (Tensor): BEV embedding sequence Q_{1:t}^m of shape (B, T, C).

        Returns:
            Tensor: aggregated spatiotemporal feature H_t^m of shape (B, C),
                taken as the last LSTM hidden state.
        """
        out, (h_t, c_t) = self.lstm(seq)
        return h_t[-1]  # (B, C)

    def spatial_interaction(self, agent_feats, map_feats=None, agent_mask=None):
        """Spatial Interaction Reasoning, Eq. (10).

        Z_t^m = CrossGNN( TempGNN(H_t^m, A), M_map )

        Args:
            agent_feats (Tensor): (B, N_agent, C) per-agent temporal features.
            map_feats (Tensor): (B, N_map, C) map / topological features.
            agent_mask (Tensor): optional (B, N_agent) padding mask.

        Returns:
            Tensor: interaction-enriched representation (B, N_agent, C).
        """
        # TempGNN: inter-agent temporal dynamics (yielding, overtaking, ...).
        h = self.temp_gnn(agent_feats, key_padding_mask=agent_mask)
        # CrossGNN: integrate topological priors (lane geometry, drivable area).
        if map_feats is not None:
            z = self.cross_gnn(h, key=map_feats, value=map_feats)
        else:
            z = self.cross_gnn(h)
        return z

    def decode(self, z):
        """Multi-modal Trajectory Decoding, Eq. (11).

        Args:
            z (Tensor): interaction-enriched representation (B, N, C).

        Returns:
            dict with:
                cls  : (B, N, M, num_maneuvers) maneuver logits (softmax over M)
                traj : (B, N, M, T_plan, 2)     predicted waypoints
                state: (B, N, M, motion_state_dim) motion-state estimates
        """
        B, N, C = z.shape
        M = self.ego_fut_mode
        # Expand into M modes via additive mode embeddings.
        zm = z.unsqueeze(2) + self.mode_embed[None, None]  # (B, N, M, C)

        cls = self.cls_head(zm)  # (B, N, M, num_maneuvers)
        cls = F.softmax(cls, dim=2)
        traj = self.reg_head(zm).view(B, N, M, self.ego_fut_ts, 2)
        state = self.stat_head(zm)  # (B, N, M, motion_state_dim)
        return dict(cls=cls, traj=traj, state=state)

    def forward(self, bev_seq, map_feats=None, agent_mask=None):
        """Full temporal-planning forward pass.

        Args:
            bev_seq (Tensor): (B, N_agent, T, C) temporal BEV embeddings per
                agent, or (B, T, C) for a single (ego) agent.
            map_feats (Tensor): optional (B, N_map, C) map features.
            agent_mask (Tensor): optional (B, N_agent) padding mask.

        Returns:
            dict of multi-modal trajectory hypotheses (see ``decode``).
        """
        if bev_seq.dim() == 3:
            bev_seq = bev_seq.unsqueeze(1)  # (B, 1, T, C)
        B, N, T, C = bev_seq.shape

        # Temporal encoding per agent (Eq. 9).
        flat = bev_seq.reshape(B * N, T, C)
        h = self.encode_temporal(flat).reshape(B, N, C)  # (B, N, C)

        # Spatial interaction reasoning (Eq. 10).
        z = self.spatial_interaction(h, map_feats, agent_mask)

        # Multi-modal decoding (Eq. 11).
        return self.decode(z)

    # ------------------------------------------------------------------ #
    def loss(self, pred, gt):
        """Planning objective, Eq. (12).

        L_plan = lambda_cls * CE(p_hat, p)
               + lambda_reg * || y_hat - y ||_1
               + lambda_stat * CE(s_hat, s)   (L1 used for continuous state)

        Args:
            pred (dict): output of ``forward`` / ``decode``.
            gt (dict): ground truth with keys
                'maneuver' (B, N), 'waypoints' (B, N, T, 2), 'state' (B, N, D),
                and optionally 'best_mode' (B, N) selecting the matched mode.

        Returns:
            dict of scalar loss terms.
        """
        cls, traj, state = pred["cls"], pred["traj"], pred["state"]
        B, N, M = cls.shape[:3]

        # Winner-take-all mode selection by min-ADE if not provided.
        if "best_mode" in gt:
            best = gt["best_mode"]  # (B, N)
        else:
            with torch.no_grad():
                err = (traj - gt["waypoints"].unsqueeze(2)).norm(dim=-1).mean(-1)
                best = err.argmin(dim=-1)  # (B, N)

        idx = best[..., None, None, None].expand(-1, -1, 1, self.ego_fut_ts, 2)
        traj_best = traj.gather(2, idx).squeeze(2)  # (B, N, T, 2)
        state_idx = best[..., None, None].expand(-1, -1, 1, self.motion_state_dim)
        state_best = state.gather(2, state_idx).squeeze(2)  # (B, N, D)

        # Maneuver classification loss (over modes).
        cls_mode = cls.mean(dim=-1)  # (B, N, M) mode confidence
        loss_cls = F.cross_entropy(
            cls_mode.reshape(B * N, M), best.reshape(B * N)
        )
        loss_reg = F.l1_loss(traj_best, gt["waypoints"])
        loss_stat = F.l1_loss(state_best, gt["state"])

        w = self.loss_weights
        total = w["cls"] * loss_cls + w["reg"] * loss_reg + w["stat"] * loss_stat
        return dict(
            loss_plan_cls=w["cls"] * loss_cls,
            loss_plan_reg=w["reg"] * loss_reg,
            loss_plan_stat=w["stat"] * loss_stat,
            loss_plan=total,
        )
