import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.core.box3d import *


###### K Frames Historical Motion and Plan Queries ######
@PLUGIN_LAYERS.register_module()
class StateQueue(nn.Module):
    def __init__(
        self,
        embed_dims,
        queue_length_motion=0,
        queue_length_plan=0,
    ):
        super(StateQueue, self).__init__()
        self.embed_dims = embed_dims
        self.queue_length_motion = queue_length_motion
        self.queue_length_plan = queue_length_plan

        self.reset()

    def reset(self):
        self.motion_query_queue = []
        self.plan_query_queue = []
        self.ego_status_feature_queue = []
        self.period = None

    def get(
        self,
        motion_query,  # torch.Size([4, 900, 6, 12, 256])
        plan_query,  # torch.Size([4, 1, 18, 6, 256])
        ego_status_feature,  # torch.Size([4, 1, 256])
        mask,  # torch.Size([4])
        temp_anchor_embed_forstate,  # torch.Size([4, 900, 2, 256])
        temp_mask_forstate,  # torch.Size([4, 900, 2])
        ego_temp_anchor_embed_forstate,  
        ego_temp_mask_forstate,  
    ):
        batch = motion_query.size(0)

        if self.period is None:
            self.reset()

            self.period = torch.zeros((batch, self.queue_length_motion),
                                      device=motion_query.device, dtype=torch.long)

            for i in range(self.queue_length_motion):
                self.motion_query_queue.append(motion_query.detach())
            
            for i in range(self.queue_length_plan):
                self.plan_query_queue.append(plan_query.detach())
                self.ego_status_feature_queue.append(ego_status_feature.detach())
        
        for i in range(batch):
            if mask is not None and mask[i] == False:
                for j in range(len(self.motion_query_queue)):
                    tmp = motion_query[i].detach()
                    self.motion_query_queue[j][i] = tmp
                
                for j in range(len(self.plan_query_queue)):
                    tmp = plan_query[i].detach()
                    self.plan_query_queue[j][i] = tmp

                for j in range(len(self.ego_status_feature_queue)):
                    tmp = ego_status_feature[i].detach()
                    self.ego_status_feature_queue[j][i] = tmp
                
                self.period[i] = 0

        temp_motion_query_forstate = torch.stack(self.motion_query_queue, dim=1)
        temp_plan_query_forstate = torch.stack(self.plan_query_queue, dim=1)
        temp_ego_status_feature_forstate = torch.stack(self.ego_status_feature_queue, dim=1)
        temp_period_forstate = self.period

        if temp_mask_forstate.size(-1) < 4:
            last_mask = temp_mask_forstate[..., 0]
            last_embed = temp_anchor_embed_forstate[..., 0, :]
            current = 4 - temp_mask_forstate.size(-1)
            if current < 3:  
                temp_motion_mask_forstate = torch.cat(
                    (torch.stack(([last_mask] * current), dim=-1), 
                     temp_mask_forstate[..., :-1]), dim=-1)
                temp_motion_embed_forstate = torch.cat(
                    (torch.stack(([last_embed] * current), dim=-2), 
                     temp_anchor_embed_forstate[..., :-1, :]), dim=-2)
            else:  
                temp_motion_mask_forstate = torch.stack(([last_mask] * current), dim=-1)
                temp_motion_embed_forstate = torch.stack(([last_embed] * current), dim=-2)
        else:
            temp_motion_mask_forstate = temp_mask_forstate[..., :-1]
            temp_motion_embed_forstate = temp_anchor_embed_forstate[..., :-1, :]
                        
        if ego_temp_mask_forstate.size(-1) < 4:
            last_mask = ego_temp_mask_forstate[..., 0]
            last_embed = ego_temp_anchor_embed_forstate[..., 0, :]
            current = 4 - ego_temp_mask_forstate.size(-1)
            if current < 3:
                temp_plan_mask_forstate = torch.cat(
                    (torch.stack(([last_mask] * current), dim=-1), 
                     ego_temp_mask_forstate[..., :-1]), dim=-1)
                temp_plan_embed_forstate = torch.cat(
                    (torch.stack(([last_embed] * current), dim=-2), 
                     ego_temp_anchor_embed_forstate[..., :-1, :]), dim=-2)
            else:
                temp_plan_mask_forstate = torch.stack(([last_mask] * current), dim=-1)
                temp_plan_embed_forstate = torch.stack(([last_embed] * current), dim=-2)
        else:
            temp_plan_mask_forstate = ego_temp_mask_forstate[..., :-1]
            temp_plan_embed_forstate = ego_temp_anchor_embed_forstate[..., :-1, :]
        
        if torch.any(temp_plan_mask_forstate):
            for i in range(batch):
                tmp_temp_plan_mask_forstate = temp_plan_mask_forstate[i]
                if torch.all(tmp_temp_plan_mask_forstate):
                    tmp_embed = ego_temp_anchor_embed_forstate[i, :, -1, :]
                    for j in range(temp_plan_embed_forstate.size(-2)):
                        temp_plan_embed_forstate[i, :, j, :] = tmp_embed
                else:
                    for j in range(temp_plan_mask_forstate.size(-1)):
                        if tmp_temp_plan_mask_forstate[..., j] == False:
                            tmp_embed = temp_plan_embed_forstate[i, :, j, :]
                            break
                    for j in range(temp_plan_mask_forstate.size(-1)):
                        if tmp_temp_plan_mask_forstate[..., j] == False:
                            break
                        temp_plan_embed_forstate[i, :, j, :] = tmp_embed
                        
        return (
            temp_motion_query_forstate, 
            temp_plan_query_forstate, 
            temp_ego_status_feature_forstate, 
            temp_period_forstate,
            temp_motion_mask_forstate.transpose(1, 2),
            temp_motion_embed_forstate.transpose(1, 2),
            temp_plan_mask_forstate.transpose(1, 2),
            temp_plan_embed_forstate.transpose(1, 2),
        )

    def cache(
        self,
        motion_query, 
        plan_query, 
        ego_status_feature,
    ):
        self.motion_query_queue.append(motion_query.detach())
        self.plan_query_queue.append(plan_query.detach())
        self.ego_status_feature_queue.append(ego_status_feature.detach())
        self.period += 1
        new_value = torch.ones((self.period.size(0), 1),
                               device=motion_query.device, dtype=torch.long)
        self.period = torch.cat((self.period, new_value), dim=1)

        if len(self.motion_query_queue) > self.queue_length_motion:
            self.motion_query_queue.pop(0)
            self.period = self.period[:, 1:]

        if len(self.plan_query_queue) > self.queue_length_plan:
            self.plan_query_queue.pop(0)
            self.ego_status_feature_queue.pop(0)

    def get_motion_for_det(
        self,
        instance_feature,  # torch.Size([3, 900, 256])
        anchor_embed,  # torch.Size([3, 900, 256])
        det_head_instance_bank_mask,  # None or torch.Size([3])
        temp_anchor_embed,  # torch.Size([3, 900, 2, 256])
        temp_mask,  # torch.Size([3, 900, 2])
    ):
        
        batch, num, dim = instance_feature.shape

        if self.period is None:
            temp_motion_query_forstate = instance_feature.unsqueeze(-2).repeat(
                1, 1, self.queue_length_motion, 1).detach()  # torch.Size([3, 900, 3, 256])
            temp_motion_mask_forstate = torch.zeros((batch, num, self.queue_length_motion),  # torch.Size([3, 900, 3])
                                                    dtype=torch.bool, device=instance_feature.device)
            temp_motion_embed_forstate = anchor_embed.unsqueeze(-2).repeat(
                1, 1, self.queue_length_motion, 1).detach()  # torch.Size([3, 900, 3, 256])
        else:
            temp_motion_query_forstate_tmp = torch.stack(self.motion_query_queue, dim=1)  # torch.Size([2, 3, 900, 6, 12, 256])
            temp_motion_query_forstate = torch.zeros(  # torch.Size([2, 3, 900, 256])
                (batch, self.queue_length_motion, num, dim),
                dtype=temp_motion_query_forstate_tmp.dtype,
                device=temp_motion_query_forstate_tmp.device
            )

            for m in range(self.period.size(0)):
                for n in range(self.period.size(1)):
                    temp_period = self.period[m][n]
                    temp_motion = temp_motion_query_forstate_tmp[m, n, :, :, temp_period-1].mean(dim=-2)
                    temp_motion_query_forstate[m][n] = temp_motion
            
            temp_motion_query_forstate = temp_motion_query_forstate.permute(0, 2, 1, 3)
            
            if temp_mask.size(-1) < 4:
                last_embed = temp_anchor_embed[..., 0, :]
                last_mask = temp_mask[..., 0]
                current = 4 - temp_mask.size(-1)
                if current < 3:  
                    temp_motion_mask_forstate = torch.cat(
                        (torch.stack(([last_mask] * current), dim=-1), 
                        temp_mask[..., :-1]), dim=-1)
                    temp_motion_embed_forstate = torch.cat(
                        (torch.stack(([last_embed] * current), dim=-2), 
                        temp_anchor_embed[..., :-1, :]), dim=-2)
                else:  
                    temp_motion_mask_forstate = torch.stack(([last_mask] * current), dim=-1)
                    temp_motion_embed_forstate = torch.stack(([last_embed] * current), dim=-2)
            else:
                temp_motion_mask_forstate = temp_mask[..., :-1]
                temp_motion_embed_forstate = temp_anchor_embed[..., :-1, :]

        # add feature in current frame
        temp_motion_query_forstate = torch.cat((temp_motion_query_forstate, 
                                                instance_feature.unsqueeze(-2).detach()),
                                               dim=-2)
        temp_motion_embed_forstate = torch.cat((temp_motion_embed_forstate, 
                                                anchor_embed.unsqueeze(-2).detach()),
                                               dim=-2)
        temp_motion_mask_forstate = torch.cat((temp_motion_mask_forstate, 
                                                torch.zeros((batch, num, 1), 
                                                            dtype=torch.bool, 
                                                            device=instance_feature.device)),
                                               dim=-1)
        
        return (
            temp_motion_query_forstate,  # torch.Size([3, 900, 4, 256])
            temp_motion_mask_forstate,  # torch.Size([3, 900, 4])
            temp_motion_embed_forstate,  # torch.Size([3, 900, 4, 256])
        )
