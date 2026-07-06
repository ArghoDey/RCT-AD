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


###### Historical Mot2Det Fusion ######
@HEADS.register_module()
class MotionforDetHead(BaseModule):
    def __init__(
        self,
        norm_layer: dict,
        ffn: dict,
        refine_layer: dict,
        decouple_attn=False,
        embed_dims=256,
        temp_graph_model=None,
        temp_graph_model_no_decouple_attn=None,
        operation_order: Optional[List[str]] = None,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(MotionforDetHead, self).__init__(init_cfg)
        
        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.embed_dims = embed_dims
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        
        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.layers_interact = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
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
    
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers_interact[i] is None:
                continue
            elif op != "refine":
                for p in self.layers_interact[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

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
            self.layers_interact[index](
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
        feature_maps,
        metas,
        det_head_anchor_encoder,  # SparseBox3DEncoder
        instance_bank,  # InstanceBank
        det_head_instance_bank_mask,  # None or torch.Size([3])
        det_head_instance_bank_anchor_handler,  # SparseBox3DKeyPointsGenerator
        motion_plan_head_instance_queue=None,  # InstanceQueue
        motion_plan_head_state_queue=None,  # StateQueue
    ):

        classification = det_output['classification']  # len=6 [torch.Size([3, 900, 10])]
        prediction = det_output['prediction']  # len=6 [torch.Size([3, 900, 11])]
        quality = det_output['quality']  # len=6 [torch.Size([3, 900, 2])]
        instance_feature = det_output['instance_feature']  # torch.Size([3, 900, 256])
        anchor_embed = det_output['anchor_embed']  # torch.Size([3, 900, 256])
        time_interval = det_output["time_interval"]  # eg: tensor([0.5000, 0.5000, 0.5000], device='cuda:0')
        anchor = det_output["anchor"]  # torch.Size([3, 900, 11])

        bs, num_anchor, dim = instance_feature.shape  # torch.Size([3, 900, 256])

        instance_queue_get = None
        if motion_plan_head_instance_queue is not None:
            (
                ego_feature,  # torch.Size([3, 1, 256])
                ego_anchor,  # torch.Size([3, 1, 11])
                temp_instance_feature,  # torch.Size([3, 901, 1, 256])
                temp_anchor,  # torch.Size([3, 901, 1, 11])
                temp_mask,  # torch.Size([3, 901, 1])
            ) = motion_plan_head_instance_queue.get(
                det_output,
                feature_maps,
                metas,
                bs,
                det_head_instance_bank_mask,
                det_head_instance_bank_anchor_handler,
            )

            instance_queue_get = (
                ego_feature,  # torch.Size([3, 1, 256])
                ego_anchor,  # torch.Size([3, 1, 11])
                temp_instance_feature,  # torch.Size([3, 901, 1, 256])
                temp_anchor,  # torch.Size([3, 901, 1, 11])
                temp_mask,  # torch.Size([3, 901, 1])
            )

            temp_anchor_embed = det_head_anchor_encoder(temp_anchor)  # torch.Size([3, 901, 1, 256])

        if motion_plan_head_state_queue is not None:
            (
                temp_motion_query_forstate,  # # torch.Size([3, 900, 4, 256])
                temp_motion_mask_forstate,  # torch.Size([3, 900, 4])
                temp_motion_embed_forstate,  # torch.Size([3, 900, 4, 256])
            ) = motion_plan_head_state_queue.get_motion_for_det(
                instance_feature,  # torch.Size([3, 900, 256])
                anchor_embed,  # torch.Size([3, 900, 256])
                det_head_instance_bank_mask,  # None or torch.Size([3])
                temp_anchor_embed[:, :num_anchor],  # torch.Size([3, 900, 2, 256])
                temp_mask[:, :num_anchor],  # torch.Size([3, 900, 2])
            )
        
        if motion_plan_head_state_queue is None:
            temp_motion_query_forstate = instance_feature.unsqueeze(-2).repeat(
                1, 1, 4, 1).detach()  # torch.Size([3, 900, 3, 256])
            temp_motion_mask_forstate = torch.zeros((bs, num_anchor, 4),  # torch.Size([3, 900, 3])
                                                    dtype=torch.bool, device=instance_feature.device)
            temp_motion_embed_forstate = anchor_embed.unsqueeze(-2).repeat(
                1, 1, 4, 1).detach()  # torch.Size([3, 900, 3, 256])
        
        output = {}

        instance_feature = instance_feature.unsqueeze(-2).reshape(
            -1, 1, dim
        )  # torch.Size([2700, 1, 256])
        temp_motion_query_forstate = temp_motion_query_forstate.reshape(
            -1, temp_motion_query_forstate.size(-2), dim
        )  # torch.Size([2700, 4, 256])
        temp_motion_embed_forstate = temp_motion_embed_forstate.reshape(
            -1, temp_motion_embed_forstate.size(-2), dim
        )  # torch.Size([2700, 4, 256])
        temp_motion_mask_forstate = temp_motion_mask_forstate.reshape(
            -1, temp_motion_mask_forstate.size(-1)
        )  # torch.Size([2700, 4])

        for i, op in enumerate(self.operation_order):
            if self.layers_interact[i] is None:
                continue
            elif op == "temp_gnn":  
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_motion_query_forstate,
                    temp_motion_query_forstate,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_motion_embed_forstate,
                    key_padding_mask=temp_motion_mask_forstate,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers_interact[i](instance_feature)
            elif op == "refine":
                instance_feature = instance_feature.squeeze(1).reshape(bs, num_anchor, dim)
                anchor, cls, qt = self.layers_interact[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=True,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)

                anchor_embed = det_head_anchor_encoder(anchor)
        
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
                "instance_id": det_output["instance_id"],
            }
        )

        return output, instance_queue_get
