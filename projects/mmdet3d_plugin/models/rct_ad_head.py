"""RCT-AD multi-task head.

Orchestrates the five concurrent RCT-AD heads on the shared BEV latent
representation (detection, map/segmentation, motion, planning, and the
Motion-for-Detection fusion), implementing the Unified Multi-Task Joint
Optimization of Section III-E:

    L_total = lambda_det  * L_det
            + lambda_seg  * L_seg
            + lambda_plan * L_plan
            + lambda_map  * L_map
            + lambda_mem  * L_mem                               (Eq. 15)

This head keeps the proven task orchestration of the baseline while exposing the
RCT-AD contributions (Temporal Trajectory Planner, Detection-and-Segmentation
fusion) through the ``motion_plan_head`` and ``motion_for_det`` slots.
"""

from typing import List, Union

import torch

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


@HEADS.register_module()
class RCTADHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        det_head=dict,
        map_head=dict,
        motion_plan_head=dict,
        motion_for_det=dict,
        init_cfg=None,
        use_motion_for_det=False,
        **kwargs,
    ):
        super(RCTADHead, self).__init__(init_cfg)
        self.task_config = task_config
        if self.task_config["with_det"]:
            self.det_head = build_head(det_head)
        if self.task_config["with_map"]:
            self.map_head = build_head(map_head)
        if self.task_config["with_motion_plan"]:
            self.motion_plan_head = build_head(motion_plan_head)

        self.use_motion_for_det = use_motion_for_det
        if self.use_motion_for_det:
            self.motion_for_det = build_head(motion_for_det)

    def init_weights(self):
        if self.task_config["with_det"]:
            self.det_head.init_weights()
        if self.task_config["with_map"]:
            self.map_head.init_weights()
        if self.task_config["with_motion_plan"]:
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if self.task_config["with_det"]:
            det_output = self.det_head(
                feature_maps, metas, use_motion_for_det=self.use_motion_for_det
            )
        else:
            det_output = None

        if self.task_config["with_map"]:
            map_output = self.map_head(feature_maps, metas)
        else:
            map_output = None

        # Motion-for-Detection Adaptive Fusion (Section III-D, Eq. 13).
        instance_queue_get = None
        if self.task_config["with_motion_plan"] and self.use_motion_for_det:
            det_output, instance_queue_get = self.motion_for_det(
                det_output,
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
                self.motion_plan_head.instance_queue,
                self.motion_plan_head.state_queue,
            )
        elif self.use_motion_for_det:
            det_output, instance_queue_get = self.motion_for_det(
                det_output,
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
            )

        if self.task_config["with_motion_plan"]:
            motion_output, planning_output = self.motion_plan_head(
                det_output,
                map_output,
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
                use_motion_for_det=self.use_motion_for_det,
                instance_queue_get=instance_queue_get,
            )
        else:
            motion_output, planning_output = None, None

        return det_output, map_output, motion_output, planning_output

    def loss(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config["with_det"]:
            losses.update(self.det_head.loss(det_output, data))

        if self.task_config["with_map"]:
            losses.update(self.map_head.loss(map_output, data))

        if self.task_config["with_motion_plan"]:
            motion_loss_cache = dict(indices=self.det_head.sampler.indices)
            losses.update(
                self.motion_plan_head.loss(
                    motion_output, planning_output, data, motion_loss_cache
                )
            )
        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, motion_output, planning_output = model_outs
        if self.task_config["with_det"]:
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)

        if self.task_config["with_map"]:
            map_result = self.map_head.post_process(map_output)
            batch_size = len(map_result)

        if self.task_config["with_motion_plan"]:
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output, motion_output, planning_output, data
            )

        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config["with_det"]:
                results[i].update(det_result[i])
            if self.task_config["with_map"]:
                results[i].update(map_result[i])
            if self.task_config["with_motion_plan"]:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])
        return results
