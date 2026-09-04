"""Multi-scale spatial-spectral encoder (paper Sec. 2.1, Eq. 1-4).

Handles the airborne H2SR image (e.g. 249 bands, 1 m).

Design (following paper + reference official implementation):
  * stem: depthwise separable conv (DSC) + two stride-2 convs  -> 1/4 resolution
  * spectral gated module (= Eq. 1-4, avg+max pooling + shared MLP + sigmoid)
  * four branches at 1/4, 1/8, 1/16, 1/32 with bottleneck conv blocks
  * cross-scale connections ONLY at the beginning (max-pool shortcuts from the
    stem into deeper branches) and at the end (upsample + add between
    branches) -- avoiding HRNet-style frequent connections (paper motivation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """DSC: depthwise 3x3 (spatial, per-band) + pointwise 1x1 (cross-band)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.point(self.depth(x))


class SpectralGatedModule(nn.Module):
    """Paper Eq. 1-4: band weights from global(avg) + local(max) statistics."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)   # global spectral features
        self.amp = nn.AdaptiveMaxPool2d(1)   # salient local features
        self.mlp = nn.Sequential(            # shared MLP (as official code)
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        w_aa = self.mlp(self.aap(x).view(b, c))   # Eq. 1
        w_am = self.mlp(self.amp(x).view(b, c))   # Eq. 2
        weights = (w_aa + w_am).view(b, c, 1, 1)  # Eq. 3
        return x * weights                        # Eq. 4


class BottleneckBlock(nn.Module):
    """1x1 -> 3x3 -> 1x1 bottleneck (same family as official BottleneckBlock)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.relu(self.bn3(self.conv3(x)))


class MultiScaleSpatialSpectralEncoder(nn.Module):
    """Returns 4 multi-scale features [1/4, 1/8, 1/16, 1/32]."""

    def __init__(self, in_bands=249, channels=(249, 300, 512, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels
        # ---- stem: DSC + 2 stride-2 convs -> 1/4 ----
        self.dsc = DepthwiseSeparableConv(in_bands, in_bands)
        self.stem = nn.Sequential(
            nn.Conv2d(in_bands, in_bands, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_bands), nn.ReLU(inplace=True),
            nn.Conv2d(in_bands, in_bands, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_bands), nn.ReLU(inplace=True),
        )
        self.gate1 = SpectralGatedModule(in_bands)

        # ---- branch 1 (1/4) ----
        self.stage1 = nn.Sequential(BottleneckBlock(in_bands, c1),
                                    BottleneckBlock(c1, c1))
        # ---- branch 2 (1/8) ----
        self.down2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.stage2 = nn.Sequential(BottleneckBlock(c2, c2),
                                    BottleneckBlock(c2, c2))
        self.dsc2 = DepthwiseSeparableConv(c2, c2)
        self.gate2 = SpectralGatedModule(c2)
        # ---- branch 3 (1/16) ----
        self.down3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True))
        self.stage3 = nn.Sequential(BottleneckBlock(c3, c3),
                                    BottleneckBlock(c3, c3))
        self.dsc3 = DepthwiseSeparableConv(c3, c3)
        self.gate3 = SpectralGatedModule(c3)
        # ---- branch 4 (1/32) ----
        self.down4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4), nn.ReLU(inplace=True))
        self.stage4 = nn.Sequential(BottleneckBlock(c4, c4),
                                    BottleneckBlock(c4, c4))
        self.dsc4 = DepthwiseSeparableConv(c4, c4)
        self.gate4 = SpectralGatedModule(c4)

        # ---- FIRST cross-scale connection: stem shortcuts into deep branches ----
        self.shortcut2 = nn.Conv2d(in_bands, c2, 1)
        self.shortcut3 = nn.Conv2d(in_bands, c3, 1)
        self.shortcut4 = nn.Conv2d(in_bands, c4, 1)

        # ---- SECOND cross-scale connection: end-of-network upsample + add ----
        self.adapter3 = nn.Sequential(nn.Conv2d(c4, c3, 1, bias=False),
                                      nn.BatchNorm2d(c3), nn.ReLU(inplace=True))
        self.adapter2 = nn.Sequential(nn.Conv2d(c3, c2, 1, bias=False),
                                      nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.adapter1 = nn.Sequential(nn.Conv2d(c2, c1, 1, bias=False),
                                      nn.BatchNorm2d(c1), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.dsc(x)
        s = self.stem(s)            # 1/4
        s = self.gate1(s)
        f1 = self.stage1(s)         # 1/4

        # beginning cross-scale: pooled stem features injected into deeper branches
        def _shortcut(conv, f, factor):
            sc = conv(F.avg_pool2d(s, factor))
            if sc.shape[-2:] != f.shape[-2:]:
                sc = F.interpolate(sc, size=f.shape[-2:], mode="bilinear",
                                   align_corners=False)
            return sc

        f2 = self.down2(f1)
        f2 = f2 + _shortcut(self.shortcut2, f2, 2)
        f2 = self.gate2(self.dsc2(f2))
        f2 = self.stage2(f2)        # 1/8

        f3 = self.down3(f2)
        f3 = f3 + _shortcut(self.shortcut3, f3, 4)
        f3 = self.gate3(self.dsc3(f3))
        f3 = self.stage3(f3)        # 1/16

        f4 = self.down4(f3)
        f4 = f4 + _shortcut(self.shortcut4, f4, 8)
        f4 = self.gate4(self.dsc4(f4))
        f4 = self.stage4(f4)        # 1/32

        # ending cross-scale: fuse deep semantics back into shallow branches
        f3 = f3 + self.adapter3(F.interpolate(f4, size=f3.shape[-2:],
                                              mode="bilinear", align_corners=False))
        f2 = f2 + self.adapter2(F.interpolate(f3, size=f2.shape[-2:],
                                              mode="bilinear", align_corners=False))
        f1 = f1 + self.adapter1(F.interpolate(f2, size=f1.shape[-2:],
                                              mode="bilinear", align_corners=False))
        return [f1, f2, f3, f4]
