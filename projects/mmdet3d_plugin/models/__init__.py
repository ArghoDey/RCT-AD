# RCT-AD model registry.
#
# Top-level RCT-AD detector and multi-task head.
from .rct_ad import RCTAD
from .rct_ad_head import RCTADHead

# Novel RCT-AD contributions.
from .rca import ReliableContextAwareness
from .det_seg import (
    DetectionSegmentationHead,
    MotionForDetectionFusion,
    SemanticBEVSegmentation,
)

# Shared building blocks inherited from the SparseDrive / BridgeAD backbone.
from .motion_for_det_head import MotionforDetHead
from .blocks import (
    DeformableFeatureAggregation,
    DenseDepthNet,
    AsymmetricFFN,
)
from .instance_bank import InstanceBank
from .detection3d import (
    SparseBox3DDecoder,
    SparseBox3DTarget,
    SparseBox3DRefinementModule,
    SparseBox3DKeyPointsGenerator,
    SparseBox3DEncoder,
)
from .map import *
from .motion import *


__all__ = [
    # RCT-AD
    "RCTAD",
    "RCTADHead",
    "ReliableContextAwareness",
    "DetectionSegmentationHead",
    "MotionForDetectionFusion",
    "SemanticBEVSegmentation",
    # backbone building blocks
    "MotionforDetHead",
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
    "InstanceBank",
    "SparseBox3DDecoder",
    "SparseBox3DTarget",
    "SparseBox3DRefinementModule",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DEncoder",
]
