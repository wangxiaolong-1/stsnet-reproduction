"""STSNet reimplementation package (paper: Information Fusion 114 (2025) 102689)."""
from .model import STSNet
from .spatial_spectral_encoder import (MultiScaleSpatialSpectralEncoder,
                                       SpectralGatedModule,
                                       DepthwiseSeparableConv)
from .spatio_temporal_encoder import (SpatioTemporalEncoder,
                                      SpatioTemporalTransformerBlock,
                                      WindowAttention3D)
from .cross_resolution_fusion import CrossResolutionFusion
from .decoder import MultiScaleDecoder
from .convlstm import ConvLSTM

__all__ = [
    "STSNet", "MultiScaleSpatialSpectralEncoder", "SpectralGatedModule",
    "DepthwiseSeparableConv", "SpatioTemporalEncoder",
    "SpatioTemporalTransformerBlock", "WindowAttention3D",
    "CrossResolutionFusion", "MultiScaleDecoder", "ConvLSTM",
]
