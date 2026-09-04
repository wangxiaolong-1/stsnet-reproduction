"""Multi-scale decoder (paper Sec. 2, component 4).

1x1 conv aligns channel numbers of each scale; fractional (transposed)
convolution upsamples to the next scale; features are fused by element-wise
addition; final 1x1 conv produces per-class logits at full resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDecoder(nn.Module):
    def __init__(self, scales_channels, num_classes, fuse_channels=128):
        """
        scales_channels: channels of encoder features ordered [1/4, 1/8, 1/16, 1/32]
        """
        super().__init__()
        self.align = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, fuse_channels, 1, bias=False),
                           nn.BatchNorm2d(fuse_channels), nn.ReLU(inplace=True))
             for c in scales_channels])
        # fractional convolutions (paper: fractional convolution for upsampling)
        self.upconv = nn.ModuleList(
            [nn.ConvTranspose2d(fuse_channels, fuse_channels, 2, stride=2)
             for _ in range(len(scales_channels))])
        self.head = nn.Conv2d(fuse_channels, num_classes, 1)

    def forward(self, feats):
        """feats: multi-scale features ordered shallow -> deep.
        The first entry (deepest = fused STSF feature) drives the bottom-up path.
        """
        # order deep -> shallow for bottom-up decoding
        feats = feats[::-1]
        n = len(feats)
        aligned = [self.align[n - 1 - i](f) for i, f in enumerate(feats)]
        x = aligned[0]
        for i in range(1, len(aligned)):
            x = self.upconv[i - 1](x)
            if x.shape[-2:] != aligned[i].shape[-2:]:
                x = F.interpolate(x, size=aligned[i].shape[-2:],
                                   mode="bilinear", align_corners=False)
            x = x + aligned[i]
        x = self.upconv[-1](x)
        # 1/4 -> full resolution (paper: interpolate to the highest resolution)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)
