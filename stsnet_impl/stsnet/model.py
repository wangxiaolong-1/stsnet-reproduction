"""STSNet: cross-spatial-resolution spatio-temporal-spectral deep fusion network.

Full model assembling the four components of the paper (Fig. 2):
  (1) multi-scale spatial-spectral encoder      -> H2SR image  (B, 249, H, W)
  (2) spatio-temporal encoder                   -> MT series   (B, T, D, h, w)
  (3) cross-resolution fusion module (alpha)    -> F_STS
  (4) multi-scale decoder                       -> logits (B, num_classes, H, W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_spectral_encoder import MultiScaleSpatialSpectralEncoder
from .spatio_temporal_encoder import SpatioTemporalEncoder
from .cross_resolution_fusion import CrossResolutionFusion
from .decoder import MultiScaleDecoder


class STSNet(nn.Module):
    def __init__(self,
                 num_classes=8,
                 h2sr_bands=249,
                 ss_channels=(249, 300, 512, 512),
                 mt_bands=10,
                 mt_times=31,
                 st_dims=(64, 96, 128),
                 st_out_channels=512,
                 alpha=0.1,
                 ws=(2, 4, 4),
                 num_heads=4):
        super().__init__()
        self.ss_encoder = MultiScaleSpatialSpectralEncoder(h2sr_bands, ss_channels)
        self.st_encoder = SpatioTemporalEncoder(
            bands=mt_bands, d2=st_dims[0], d3=st_dims[1], d4=st_dims[2],
            out_channels=st_out_channels, num_times=mt_times,
            ws=ws, num_heads=num_heads)
        self.fusion = CrossResolutionFusion(channels=ss_channels[-1], alpha=alpha)
        self.decoder = MultiScaleDecoder(list(ss_channels), num_classes)
        # channel adapter if ST branch output differs from deepest SS channels
        self.st_adapter = (nn.Conv2d(st_out_channels, ss_channels[-1], 1, bias=False)
                           if st_out_channels != ss_channels[-1] else nn.Identity())

    def forward(self, x_h2sr, x_mt):
        """x_h2sr: (B, bands, H, W);  x_mt: (B, T, bands, h, w)."""
        feats = self.ss_encoder(x_h2sr)          # [1/4, 1/8, 1/16, 1/32]
        f_ss = feats[-1]
        f_st = self.st_encoder(x_mt)             # (B, C_st, h, w)
        # spatially resample ST features to SS semantic feature size
        f_st = self.st_adapter(
            F.interpolate(f_st, size=f_ss.shape[-2:],
                          mode="bilinear", align_corners=False))
        f_sts = self.fusion(f_ss, f_st)          # Eq. 11-12
        return self.decoder([*feats[:-1], f_sts])
