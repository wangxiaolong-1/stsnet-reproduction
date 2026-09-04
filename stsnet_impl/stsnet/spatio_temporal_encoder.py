"""Spatio-temporal encoder (paper Sec. 2.2). This part is MISSING from the
official repo, implemented from scratch following the paper.

Pipeline:
  input (B, T, D, H, W)  e.g. (B, 31, 10, 30, 30) Sentinel-2 dense series
    | spectral convolution (3D conv over spectral dim + skip connections)
    | 3D patch partition (fold 2x2x2 voxels into channels, Swin-style)
    | linear embedding (1x1x1 conv)
    | spatio-temporal transformer block #1  (ConvLSTM  ||  3D W-MSA / SW-MSA)
    | patch merging (halve T,H,W, double channels)
    | spatio-temporal transformer block #2 (with shortcut from block #1)
    | upsample back, temporal-to-channel fold, 1x1 conv
  output (B, out_channels, H, W)  2-D semantic feature of the MT modality

Note: the paper's tensor-shape notation for patch partition is ambiguous
("(B, D2, T/2, H/2, W/2)"); we adopt the standard Swin convention of folding
the 2x2x2 patch into the channel dimension, which is dimensionally consistent
with the rest of the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm import ConvLSTM


class SpectralConv(nn.Module):
    """3D convolution along the spectral dimension with skip connections.

    Input/output: (B, T, D_in, H, W) -> (B, T, D_out, H, W)
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv = nn.Conv3d(d_in, d_out, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.skip = (nn.Conv3d(d_in, d_out, kernel_size=1)
                     if d_in != d_out else nn.Identity())

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))


class PatchPartition3D(nn.Module):
    """Fold (2,2,2) voxel patches into channels: (B,C,T,H,W)->(B,8C,T/2,H/2,W/2)."""

    def __init__(self, patch=(2, 2, 2)):
        super().__init__()
        self.p = patch

    def forward(self, x):
        pt, ph, pw = self.p
        B, C, T, H, W = x.shape
        # pad dims to patch multiples (e.g. T=31 -> 32 for patch size 2)
        pad_t, pad_h, pad_w = (-T) % pt, (-H) % ph, (-W) % pw
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
            T, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.reshape(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
        # (B, C, pt, ph, pw, T', H', W')
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return x.reshape(B, C * pt * ph * pw, T // pt, H // ph, W // pw)


class PatchMerging3D(nn.Module):
    """Downscale T,H,W by 2 and double channels (Swin patch merging, 3D).

    A 2x2x2 voxel neighborhood folds into 8*ch_in features, then a linear
    layer reduces to ch_out (default 2*ch_in).
    """

    def __init__(self, ch_in, ch_out=None, norm=True):
        super().__init__()
        ch_out = ch_out or 2 * ch_in
        self.norm = nn.LayerNorm(8 * ch_in) if norm else nn.Identity()
        self.reduction = nn.Linear(8 * ch_in, ch_out, bias=False)

    def forward(self, x):
        # x: (B, C, T, H, W) channel-first for conv layers
        B, C, T, H, W = x.shape
        # pad to even sizes in T, H, W
        pad_t, pad_h, pad_w = (-T) % 2, (-H) % 2, (-W) % 2
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
            T, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 3, 4, 1)                      # (B,T,H,W,C)
        T2, H2, W2 = T // 2, H // 2, W // 2
        x = x.reshape(B, T2, 2, H2, 2, W2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # (B,T2,H2,W2,2,2,2,C)
        x = x.reshape(B, T2, H2, W2, 8 * C)
        x = self.reduction(self.norm(x))                  # (B,T2,H2,W2,C_out)
        return x.permute(0, 4, 1, 2, 3).contiguous()      # (B,C_out,T2,H2,W2)


def window_partition3d(x, ws):
    """x: (B, C, T, H, W) -> (num_windows*B, C, wt*wh*ww tokens dim moves later)."""
    wt, wh, ww = ws
    B, C, T, H, W = x.shape
    x = x.reshape(B, C, T // wt, wt, H // wh, wh, W // ww, ww)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()   # (B,nT,nH,nW,C,wt,wh,ww)
    return x.reshape(-1, C, wt, wh, ww)


def window_reverse3d(x, ws, B, T, H, W):
    wt, wh, ww = ws
    x = x.reshape(B, T // wt, H // wh, W // ww, -1, wt, wh, ww)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.reshape(B, -1, T, H, W)


class WindowAttention3D(nn.Module):
    """3D W-MSA / 3D SW-MSA (paper Eq. 10) over (T,H,W) windows.

    Input (B, C, T, H, W) channel-first; attention runs on tokens of dim C.
    """

    def __init__(self, dim, ws=(2, 4, 4), num_heads=4, shifted=False):
        super().__init__()
        self.dim, self.ws, self.shifted = dim, ws, shifted
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        wt, wh, ww = self.ws
        B, C, T, H, W = x.shape
        # pad to window multiples
        pT, pH, pW = (-T) % wt, (-H) % wh, (-W) % ww
        if pT or pH or pW:
            x = F.pad(x, (0, pW, 0, pH, 0, pT))
        if self.shifted:
            x = torch.roll(x, shifts=(-wt // 2, -wh // 2, -ww // 2), dims=(2, 3, 4))
        Tp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]

        x = window_partition3d(x, self.ws)               # (nW*B, C, wt, wh, ww)
        nwB, C_, _, _, _ = x.shape
        tokens = x.reshape(nwB, C_, -1).permute(0, 2, 1)  # (nW*B, N, C)

        qkv = self.qkv(tokens).reshape(nwB, -1, 3, self.num_heads, C_ // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)    # (nW*B, heads, N, d)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)                       # Eq. 10
        out = (attn @ v).transpose(1, 2).reshape(nwB, -1, C_)
        out = self.proj(out).permute(0, 2, 1).reshape(nwB, C_, wt, wh, ww)

        out = window_reverse3d(out, self.ws, B, Tp, Hp, Wp)
        if self.shifted:
            out = torch.roll(out, shifts=(wt // 2, wh // 2, ww // 2), dims=(2, 3, 4))
        if pT or pH or pW:
            out = out[:, :, :T, :H, :W].contiguous()
        return out


class SpatioTemporalTransformerBlock(nn.Module):
    """LN + [ConvLSTM (local) || 3D W-MSA + 3D SW-MSA (global)] + MLP.

    Paper Sec. 2.2: local and global spatio-temporal information are extracted
    in PARALLEL, both with residual connections.
    """

    def __init__(self, dim, hidden_dim, ws=(2, 4, 4), num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.convlstm = ConvLSTM(dim, dim)
        self.wmsa = WindowAttention3D(dim, ws, num_heads, shifted=False)
        self.swmsa = WindowAttention3D(dim, ws, num_heads, shifted=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def _ncl(self, x):
        return x.permute(0, 2, 3, 4, 1)  # (B,C,T,H,W)->(B,T,H,W,C)

    def _cln(self, x):
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x):
        # local branch: ConvLSTM over T (needs (B,T,C,H,W))
        xl = self._ncl(x)
        xl = self.norm1(xl).permute(0, 1, 4, 2, 3)             # (B,T,C,H,W)
        xl = self.convlstm(xl)
        xl = xl.permute(0, 2, 1, 3, 4)                           # (B,C,T,H,W)
        # global branch: 3D window attention (needs channel-last for LN)
        xg = self._ncl(x)
        xg = self._cln(self.attn_norm(xg))
        xg = self.wmsa(xg) + self.swmsa(xg)
        # parallel fusion + MLP
        out = x + xl + xg
        out = self._ncl(out)
        out = out + self.mlp(self.norm2(out))
        return self._cln(out)


class SpatioTemporalEncoder(nn.Module):
    """Full encoder for dense time-series multispectral imagery."""

    def __init__(self, bands=10, d2=64, d3=96, d4=128, out_channels=512,
                 num_times=31, ws=(2, 4, 4), num_heads=4):
        super().__init__()
        self.num_times = num_times
        self.spectral_conv = SpectralConv(bands, d2)
        self.patch_partition = PatchPartition3D((2, 2, 2))
        self.linear_embed = nn.Conv3d(d2 * 8, d3, kernel_size=1)
        self.block1 = SpatioTemporalTransformerBlock(d3, 2 * d3, ws, num_heads)
        self.patch_merging = PatchMerging3D(d3, d4)
        self.block2 = SpatioTemporalTransformerBlock(d4, 2 * d4, ws, num_heads)
        # conv adjusts after block2 (paper: 2*D4 with shortcut from block1)
        self.conv_after_b2 = nn.Conv3d(d4, 2 * d4, kernel_size=1)
        self.shortcut_proj = nn.Conv3d(d3, 2 * d4, kernel_size=1)
        self.conv_out = nn.Conv3d(2 * d4, d2, kernel_size=1)
        # output head: input channels = D2 * T (temporal fold over the whole series)
        self.head = nn.Conv2d(d2 * num_times, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, T, D, H, W) -> (B, D, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.spectral_conv(x)                       # (B, D2, T, H, W)
        x = self.patch_partition(x)                     # (B, 8*D2, T/2, H/2, W/2)
        x = self.linear_embed(x)                        # (B, D3, ...)

        y1 = self.block1(x)                             # (B, D3, T/2, ...)
        x = self.patch_merging(y1)                      # (B, D4, T/4, ...)
        y2 = self.block2(x)
        y2 = F.relu(self.conv_after_b2(y2) +
                    self.shortcut_proj(F.adaptive_avg_pool3d(y1, y2.shape[2:])))
        y2 = F.interpolate(y2, size=y1.shape[2:], mode="trilinear", align_corners=False)
        y2 = F.relu(self.conv_out(y2))                  # (B, D2, T/2, H/2, W/2)
        y2 = F.interpolate(y2, scale_factor=2, mode="trilinear", align_corners=False)

        # dimension transformation: (B, D2, T, H, W) -> (B, D2*T, H, W)
        B, C, T, H, W = y2.shape
        y2 = y2.permute(0, 1, 2, 3, 4).reshape(B, C * T, H, W)
        return self.head(y2)                            # (B, out_channels, H, W)
