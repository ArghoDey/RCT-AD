from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


class PlanningRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=6,
    ):
        super(PlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 2),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        plan_query,  # torch.Size([6, 1, 18, 256]) torch.Size([6, 1, 18, 6, 256])
        ego_feature,  # torch.Size([6, 1, 256])
        ego_anchor_embed,  # torch.Size([6, 1, 256])
    ):
        bs, num_anchor = plan_query.shape[:2]
        
        plan_cls = self.plan_cls_branch(plan_query.mean(dim=-2)).squeeze(-1)
        
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
        
        ego_status_feature = ego_feature + ego_anchor_embed
        planning_status = self.plan_status_branch(ego_status_feature)

        # torch.Size([6, 1, 18]) torch.Size([6, 1, 18, 6, 2]) torch.Size([6, 1, 10])
        return plan_cls, plan_reg, planning_status, ego_status_feature
    

@HEADS.register_module()
class MotionPlanningHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        state_queue=None,
        operation_order=None,
        operation_order_state_self_motion=None,
        operation_order_state_self_plan=None,
        operation_order_state_motion=None,
        operation_order_state_plan=None,
        operation_order_state_motplan=None,
        temp_graph_model=None,
        temp_graph_model_no_decouple_attn=None,
        temp_graph_model_no_decouple_flash_attn=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
    ):
        super(MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.operation_order_state_self_motion = operation_order_state_self_motion
        self.operation_order_state_self_plan = operation_order_state_self_plan  
        self.operation_order_state_motion = operation_order_state_motion
        self.operation_order_state_plan = operation_order_state_plan
        self.operation_order_state_motplan = operation_order_state_motplan
        self.embed_dims = embed_dims

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.state_queue = build(state_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }

        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.layers_state_motion = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order_state_motion
            ]
        )
        self.layers_state_plan = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order_state_plan
            ]
        )
        self.layers_state_motion_cross_state = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order_state_self_motion
            ]
        )
        self.layers_state_motion_cross_mode = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order_state_self_motion
            ]
        )
        self.layers_state_plan_cross_state = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order_state_self_plan
            ]
        )
        self.layers_state_plan_cross_mode = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order_state_self_plan
            ]
        )
        self.layers_state_motion_with_plan = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order_state_motplan
            ]
        )

        if "temp_gnn" in self.operation_order_state_motion:
            self.fc_before_state_motion = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after_state_motion = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )

        if "gnn" in self.operation_order_state_plan:
            self.fc_before_state_plan = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after_state_plan = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )

        if "gnn" in self.operation_order_state_self_motion:
            self.fc_before_self_motion = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after_self_motion = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()
        
        self.planningrefine = PlanningRefinementModule()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order_state_motion):
            if self.layers_state_motion[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_state_motion[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order_state_plan):
            if self.layers_state_plan[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_state_plan[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order_state_self_motion):
            if self.layers_state_motion_cross_state[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_state_motion_cross_state[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order_state_self_motion):
            if self.layers_state_motion_cross_mode[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_state_motion_cross_mode[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order_state_self_plan):
            if self.layers_state_plan_cross_state[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_state_plan_cross_state[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order_state_self_plan):
            if self.layers_state_plan_cross_mode[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_state_plan_cross_mode[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order_state_motplan):
            if self.layers_state_motion_with_plan[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_state_motion_with_plan[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def graph_model_state_motion(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before_state_motion(value)
        return self.fc_after_state_motion(
            self.layers_state_motion[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def graph_model_state_plan(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before_state_plan(value)
        return self.fc_after_state_plan(
            self.layers_state_plan[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def graph_model_self_motion_state(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before_self_motion(value)
        return self.fc_after_self_motion(
            self.layers_state_motion_cross_state[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )
    
    def graph_model_self_motion_mode(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before_self_motion(value)
        return self.fc_after_self_motion(
            self.layers_state_motion_cross_mode[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )
    
    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
        use_motion_for_det=False,
        instance_queue_get=None,
    ):   
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]  # torch.Size([6, 900, 256])
        anchor_embed = det_output["anchor_embed"]  # torch.Size([6, 900, 256])
        det_classification = det_output["classification"][-1].sigmoid()  # torch.Size([6, 900, 10])
        det_anchors = det_output["prediction"][-1]  # torch.Size([6, 900, 11])
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )  # self.num_det = 50

        map_instance_feature = map_output["instance_feature"]  # torch.Size([6, 100, 256])
        map_anchor_embed = map_output["anchor_embed"]  # torch.Size([6, 100, 256])
        map_classification = map_output["classification"][-1].sigmoid()  # torch.Size([6, 100, 3])
        map_anchors = map_output["prediction"][-1]  # torch.Size([6, 100, 40])
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )  # self.num_map = 10

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape  # torch.Size([6, 900, 256])
        if not use_motion_for_det:
            (
                ego_feature,  # torch.Size([6, 1, 256])
                ego_anchor,  # torch.Size([6, 1, 11])
                temp_instance_feature,  # torch.Size([6, 901, 1, 256])
                temp_anchor,  # torch.Size([6, 901, 1, 11])
                temp_mask,  # torch.Size([6, 901, 1])
            ) = self.instance_queue.get(
                det_output,
                feature_maps,
                metas,
                bs,
                mask,
                anchor_handler,
            )
        else:
            (
                ego_feature,  # torch.Size([6, 1, 256])
                ego_anchor,  # torch.Size([6, 1, 11])
                temp_instance_feature,  # torch.Size([6, 901, 1, 256])
                temp_anchor,  # torch.Size([6, 901, 1, 11])
                temp_mask,  # torch.Size([6, 901, 1])
            ) = instance_queue_get
        
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_anchor_embed_forstate = temp_anchor_embed  # torch.Size([4, 901, 2, 256])
        temp_mask_forstate = temp_mask  # torch.Size([4, 901, 2])
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)  # torch.Size([6, 900, 6, 12, 2])
        plan_anchor = torch.tile(  # torch.Size([6, 3, 6, 6, 2])
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor))  # torch.Size([6, 900, 6, 256])
        plan_pos = gen_sineembed_for_position(plan_anchor)
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)  # torch.Size([6, 1, 18, 256])

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)  # torch.Size([6, 51, 256])
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)  # torch.Size([6, 51, 256])

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)  # torch.Size([6, 901, 256])
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)  # torch.Size([6, 901, 256])

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":  # Agent-Temporal Cross Attention
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":  # Agent-Agent Self Attention
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":  # Agent-Map Cross Attention
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2).unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2).unsqueeze(2)
                
                (
                    temp_motion_query_forstate,  # torch.Size([4, 3, 900, 6, 12, 256])
                    temp_plan_query_forstate,  # torch.Size([4, 3, 1, 18, 6, 256])
                    temp_ego_status_feature_forstate,  # torch.Size([4, 3, 1, 256])
                    temp_period_forstate,  # torch.Size([4, 3])
                    temp_motion_mask_forstate,  # torch.Size([4, 3, 900])
                    temp_motion_embed_forstate,  # torch.Size([4, 3, 900, 256])
                    temp_plan_mask_forstate,  # torch.Size([4, 3, 1])
                    temp_plan_embed_forstate,  # torch.Size([4, 3, 1, 256])
                ) = self.state_queue.get(
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:] + anchor_embed[:, num_anchor:],
                    mask,
                    temp_anchor_embed_forstate[:, :num_anchor],  # torch.Size([4, 900, 2, 256])
                    temp_mask_forstate[:, :num_anchor],  # torch.Size([4, 900, 2])
                    temp_anchor_embed_forstate[:, num_anchor:],  
                    temp_mask_forstate[:, num_anchor:],  
                )

                ###### 1.History-Enhanced Motion Prediction ######
                interact_state = 6
                shapes = list(temp_motion_query_forstate.shape)
                shapes[-2] = interact_state
                temp_motion_query_forstate_selected = torch.zeros(
                    shapes, 
                    device=temp_motion_query_forstate.device, 
                    dtype=temp_motion_query_forstate.dtype,
                )  # torch.Size([4, 3, 900, 6, 9, 256])

                for m in range(temp_period_forstate.size(0)):
                    for n in range(temp_period_forstate.size(1)):
                        temp_period = temp_period_forstate[m][n]
                        temp_motion = temp_motion_query_forstate[m, n, ..., temp_period:temp_period+interact_state, :]
                        temp_motion_query_forstate_selected[m][n] = temp_motion
                
                motion_query_new = motion_query[..., :interact_state, :]  # torch.Size([4, 900, 6, 9, 256])

                batch, num, mode, time, dim = motion_query.shape
                
                motion_query_new = motion_query_new.unsqueeze(-2)  
                # torch.Size([4, 900, 6, 9, 1, 256])
                temp_motion_query_forstate_selected = temp_motion_query_forstate_selected.permute(0, 2, 3, 4, 1, 5)
                # torch.Size([4, 900, 6, 9, 3, 256])
                temp_motion_mask_forstate = temp_motion_mask_forstate.permute(0, 2, 1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, mode, interact_state, 1)
                # torch.Size([4, 900, 6, 9, 3])
                temp_motion_embed_forstate = temp_motion_embed_forstate.permute(0, 2, 1, 3).unsqueeze(-3).unsqueeze(-3).repeat(1, 1, mode, interact_state, 1, 1)
                # torch.Size([4, 900, 6, 9, 3, 256])
                
                # add current to temp to avoid NaN
                current_motion_query_forstate_selected = motion_query_new.detach()
                current_motion_mask_forstate = torch.zeros(
                    current_motion_query_forstate_selected.shape[:-1],
                    device=current_motion_query_forstate_selected.device, 
                    dtype=torch.bool,
                )
                current_motion_embed_forstate = anchor_embed[:, :num_anchor].unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, mode, interact_state, 1, 1)
                
                temp_motion_query_forstate_selected = torch.cat((temp_motion_query_forstate_selected, current_motion_query_forstate_selected), dim=-2)
                temp_motion_mask_forstate = torch.cat((temp_motion_mask_forstate, current_motion_mask_forstate), dim=-1)
                temp_motion_embed_forstate = torch.cat((temp_motion_embed_forstate, current_motion_embed_forstate), dim=-2)

                motion_query_new = motion_query_new.reshape(-1, motion_query_new.size(-2), motion_query_new.size(-1))
                temp_motion_query_forstate_selected = temp_motion_query_forstate_selected.reshape(-1, temp_motion_query_forstate_selected.size(-2), temp_motion_query_forstate_selected.size(-1))
                temp_motion_mask_forstate = temp_motion_mask_forstate.reshape(-1, temp_motion_mask_forstate.size(-1))
                temp_motion_embed_forstate = temp_motion_embed_forstate.reshape(-1, temp_motion_embed_forstate.size(-2), temp_motion_embed_forstate.size(-1))

                for j, ops in enumerate(self.operation_order_state_motion):
                    if self.layers_state_motion[j] is None:
                        continue
                    elif ops == "norm" or ops == "ffn":
                        motion_query_new = self.layers_state_motion[j](motion_query_new)
                    elif ops == "temp_gnn":  
                        motion_query_new = self.graph_model_state_motion(
                            j,
                            motion_query_new,
                            temp_motion_query_forstate_selected,
                            temp_motion_query_forstate_selected,
                            query_pos=anchor_embed[:, :num_anchor].unsqueeze(-2).reshape(-1, 1, dim).repeat(mode*interact_state, 1, 1),
                            key_pos=temp_motion_embed_forstate,
                            key_padding_mask=temp_motion_mask_forstate,
                        )
                motion_query_new = motion_query_new.reshape(batch, num, mode, interact_state, motion_query_new.size(-2), motion_query_new.size(-1)).squeeze(-2)
                motion_query[..., :interact_state, :] = motion_query_new

                motion_query = motion_query.reshape(-1, time, dim)
                for j, ops in enumerate(self.operation_order_state_self_motion):
                    if self.layers_state_motion_cross_state[j] is None:
                        continue
                    elif ops == "norm" or ops == "ffn":
                        motion_query = self.layers_state_motion_cross_state[j](motion_query)
                    elif ops == "cross_gnn":  
                        motion_query = self.layers_state_motion_cross_state[j](
                            motion_query,
                        )
                    elif ops == "gnn": 
                        motion_query = self.graph_model_self_motion_state(
                            j,
                            motion_query,
                            motion_query,
                            motion_query,
                            query_pos=anchor_embed[:, :num_anchor].unsqueeze(-2).unsqueeze(-2).repeat(1, 1, mode, time, 1).reshape(-1, time, dim),
                            key_pos=anchor_embed[:, :num_anchor].unsqueeze(-2).unsqueeze(-2).repeat(1, 1, mode, time, 1).reshape(-1, time, dim),
                        )
                motion_query = motion_query.reshape(batch, num, mode, time, dim)
                motion_query = motion_query.transpose(2,3).reshape(-1, mode, dim)
                for j, ops in enumerate(self.operation_order_state_self_motion):
                    if self.layers_state_motion_cross_mode[j] is None:
                        continue
                    elif ops == "norm" or ops == "ffn":
                        motion_query = self.layers_state_motion_cross_mode[j](motion_query)
                    elif ops == "cross_gnn":  
                        motion_query = self.layers_state_motion_cross_mode[j](
                            motion_query,
                        )
                    elif ops == "gnn": 
                        motion_query = self.graph_model_self_motion_mode(
                            j,
                            motion_query,
                            motion_query,
                            motion_query,
                            query_pos=anchor_embed[:, :num_anchor].unsqueeze(-2).unsqueeze(-2).repeat(1, 1, time, mode, 1).reshape(-1, mode, dim),
                            key_pos=anchor_embed[:, :num_anchor].unsqueeze(-2).unsqueeze(-2).repeat(1, 1, time, mode, 1).reshape(-1, mode, dim),
                        )
                motion_query = motion_query.reshape(batch, num, time, mode, dim).transpose(2,3)
                
                _, num_plan, mode_plan, time_plan, _ = plan_query.shape

                ###### 2.History-Enhanced Planning ######
                interact_state_plan = 3
                shapes = list(temp_plan_query_forstate.shape)
                shapes[-2] = interact_state_plan

                temp_plan_query_forstate_selected = torch.zeros(
                    shapes, 
                    device=temp_plan_query_forstate.device, 
                    dtype=temp_plan_query_forstate.dtype,
                )  # torch.Size([4, 3, 1, 18, 3, 256])

                for m in range(temp_period_forstate.size(0)):
                    for n in range(temp_period_forstate.size(1)):
                        temp_period = temp_period_forstate[m][n]
                        temp_plan = temp_plan_query_forstate[m, n, ..., temp_period:temp_period+interact_state_plan, :]
                        temp_plan_query_forstate_selected[m][n] = temp_plan

                plan_query_new = plan_query[..., :interact_state_plan, :]  # torch.Size([4, 1, 18, 3, 256])

                plan_query_new = plan_query_new.unsqueeze(-2)  
                # torch.Size([4, 1, 18, 3, 1, 256])
                temp_plan_query_forstate_selected = temp_plan_query_forstate_selected.permute(0, 2, 3, 4, 1, 5)
                # torch.Size([4, 1, 18, 3, 3, 256])
                temp_plan_mask_forstate = temp_plan_mask_forstate.permute(0, 2, 1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, mode_plan, interact_state_plan, 1)
                # torch.Size([4, 1, 18, 3, 3])
                temp_plan_embed_forstate = temp_plan_embed_forstate.permute(0, 2, 1, 3).unsqueeze(-3).unsqueeze(-3).repeat(1, 1, mode_plan, interact_state_plan, 1, 1)
                # torch.Size([4, 1, 18, 3, 3, 256])

                # add current to temp to avoid NaN
                current_plan_query_forstate_selected = plan_query_new.detach()
                current_plan_mask_forstate = torch.zeros(
                    current_plan_query_forstate_selected.shape[:-1],
                    device=current_plan_query_forstate_selected.device, 
                    dtype=torch.bool,
                )
                current_plan_embed_forstate = anchor_embed[:, num_anchor:].unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, mode_plan, interact_state_plan, 1, 1)
                
                temp_plan_query_forstate_selected = torch.cat((temp_plan_query_forstate_selected, current_plan_query_forstate_selected), dim=-2)
                temp_plan_mask_forstate = torch.cat((temp_plan_mask_forstate, current_plan_mask_forstate), dim=-1)
                temp_plan_embed_forstate = torch.cat((temp_plan_embed_forstate, current_plan_embed_forstate), dim=-2)

                plan_query_new = plan_query_new.reshape(-1, plan_query_new.size(-2), plan_query_new.size(-1))
                temp_plan_query_forstate_selected = temp_plan_query_forstate_selected.reshape(-1, temp_plan_query_forstate_selected.size(-2), temp_plan_query_forstate_selected.size(-1))
                temp_plan_mask_forstate = temp_plan_mask_forstate.reshape(-1, temp_plan_mask_forstate.size(-1))
                temp_plan_embed_forstate = temp_plan_embed_forstate.reshape(-1, temp_plan_embed_forstate.size(-2), temp_plan_embed_forstate.size(-1))

                for j, ops in enumerate(self.operation_order_state_plan):
                    if self.layers_state_plan[j] is None:
                        continue
                    elif ops == "norm" or ops == "ffn":
                        plan_query_new = self.layers_state_plan[j](plan_query_new)
                    elif ops == "gnn":  
                        plan_query_new = self.graph_model_state_plan(
                            j,
                            plan_query_new,
                            temp_plan_query_forstate_selected,
                            temp_plan_query_forstate_selected,
                            query_pos=anchor_embed[:, num_anchor:].unsqueeze(-2).reshape(-1, 1, dim).repeat(mode_plan*interact_state_plan, 1, 1),
                            key_pos=temp_plan_embed_forstate,
                        )
                
                plan_query_new = plan_query_new.reshape(batch, num_plan, mode_plan, interact_state_plan, plan_query_new.size(-2), plan_query_new.size(-1)).squeeze(-2)
                plan_query[..., :interact_state_plan, :] = plan_query_new  # torch.Size([4, 1, 18, 6, 256])
                
                plan_query = plan_query.reshape(-1, time_plan, dim)
                for j, ops in enumerate(self.operation_order_state_self_plan):
                    if self.layers_state_plan_cross_state[j] is None:
                        continue
                    elif ops == "norm" or ops == "ffn":
                        plan_query = self.layers_state_plan_cross_state[j](plan_query)
                    elif ops == "cross_gnn":  
                        plan_query = self.layers_state_plan_cross_state[j](
                            plan_query,
                        )
                plan_query = plan_query.reshape(batch, num_plan, mode_plan, time_plan, dim)

                plan_query = plan_query.transpose(2,3).reshape(-1, mode_plan, dim)
                for j, ops in enumerate(self.operation_order_state_self_plan):
                    if self.layers_state_plan_cross_mode[j] is None:
                        continue
                    elif ops == "norm" or ops == "ffn":
                        plan_query = self.layers_state_plan_cross_mode[j](plan_query)
                    elif ops == "cross_gnn":  
                        plan_query = self.layers_state_plan_cross_mode[j](
                            plan_query,
                        )
                plan_query = plan_query.reshape(batch, num_plan, time_plan, mode_plan, dim).transpose(2,3)
                
                ###### motion refinement module ######
                (
                    motion_cls,  # torch.Size([6, 900, 6])
                    motion_reg,  # torch.Size([6, 900, 6, 12, 2])
                    # plan_cls,  # torch.Size([6, 1, 18])
                    # plan_reg,  # torch.Size([6, 1, 18, 6, 2])
                    # plan_status,  # torch.Size([6, 1, 10])
                    # ego_status_feature,  # torch.Size([6, 1, 256])
                ) = self.layers[i](
                    motion_query,  # torch.Size([6, 900, 6, 256]) torch.Size([6, 900, 6, 12, 256])
                    # plan_query,  # torch.Size([6, 1, 18, 256]) torch.Size([6, 1, 18, 6, 256])
                    # instance_feature[:, num_anchor:],
                    # anchor_embed[:, num_anchor:],
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)

                ###### 3.Step-Level Mot2Plan Interaction ######
                motion_probabilities = motion_cls.sigmoid().detach()
                max_prob_indices = torch.argmax(motion_probabilities, dim=2)
                max_prob_indices = max_prob_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(bs, num, 1, time, dim)
                selected_motion_query = torch.gather(motion_query.detach(), 2, max_prob_indices)

                plan_query = plan_query.permute(0, 3, 1, 2, 4).reshape(batch*time_plan, -1, dim)
                motion_query_for_plan = selected_motion_query[..., :time_plan, :].permute(0, 3, 1, 2, 4).reshape(batch*time_plan, -1, dim)
                for j, ops in enumerate(self.operation_order_state_motplan):
                    if self.layers_state_motion_with_plan[j] is None:
                        continue
                    elif ops == "norm" or ops == "ffn":
                        plan_query = self.layers_state_motion_with_plan[j](plan_query)
                    elif ops == "cross_gnn":  
                        plan_query = self.layers_state_motion_with_plan[j](
                            plan_query,
                            key=motion_query_for_plan,
                        )
                plan_query = plan_query.reshape(batch, time_plan, num_plan, mode_plan, dim).permute(0, 2, 3, 1, 4)

                ###### plan refinement module ######
                (
                    # motion_cls,  # torch.Size([6, 900, 6])
                    # motion_reg,  # torch.Size([6, 900, 6, 12, 2])
                    plan_cls,  # torch.Size([6, 1, 18])
                    plan_reg,  # torch.Size([6, 1, 18, 6, 2])
                    plan_status,  # torch.Size([6, 1, 10])
                    ego_status_feature,  # torch.Size([6, 1, 256])
                ) = self.planningrefine(
                    # motion_query,  # torch.Size([6, 900, 6, 256]) torch.Size([6, 900, 6, 12, 256])
                    plan_query,  # torch.Size([6, 1, 18, 256]) torch.Size([6, 1, 18, 6, 256])
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        self.state_queue.cache(motion_query, plan_query, ego_status_feature)

        motion_output = {
            "classification": motion_classification,  # torch.Size([6, 900, 6])
            "prediction": motion_prediction,  # torch.Size([6, 900, 6, 12, 2])
            "period": self.instance_queue.period,  # torch.Size([6, 900])
            "anchor_queue": self.instance_queue.anchor_queue,  # torch.Size([6, 900, 11])
        }
        planning_output = {
            "classification": planning_classification,  # torch.Size([6, 1, 18])
            "prediction": planning_prediction,  # torch.Size([6, 1, 18, 6, 2])
            "status": planning_status,  # torch.Size([6, 1, 10])
            "period": self.instance_queue.ego_period,  # torch.Size([6, 1])
            "anchor_queue": self.instance_queue.ego_anchor_queue,  # torch.Size([6, 1, 11])
        }
        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]  # torch.Size([6, 900, 6])
        reg_preds = model_outs["prediction"]  # torch.Size([6, 900, 6, 12, 2])
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target,  # torch.Size([6, 900])
                cls_weight,  # torch.Size([6, 900])
                reg_pred,  # torch.Size([6, 900, 12, 2])
                reg_target,  # torch.Size([6, 900, 12, 2])
                reg_weight,  # torch.Size([6, 900, 12])
                num_pos  # torch.Size([1])
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]  # torch.Size([6, 1, 18])
        reg_preds = model_outs["prediction"]  # torch.Size([6, 1, 18, 6, 2])
        status_preds = model_outs["status"]  # torch.Size([6, 1, 10])
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,  # torch.Size([6, 1, 6])
                cls_target,  # torch.Size([6, 1])
                cls_weight,  # torch.Size([6, 1])
                reg_pred,  # torch.Size([6, 1, 6, 2])
                reg_target,  # torch.Size([6, 1, 6, 2])
                reg_weight,  # torch.Size([6, 1, 6])
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result
    