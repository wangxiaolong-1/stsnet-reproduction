"""Cross-resolution fusion module (paper Sec. 2.3, Eq. 11-12). Core contribution.

Fuses the deepest semantic features of both modalities:
  F_SS : spatial-spectral features from H2SR encoder     (B, C, H, W)
  F_ST : spatio-temporal features from MT encoder, already resampled
         to the same spatial size                         (B, C, H, W)

Eq. 11:  F_STS = F_SS + alpha * ( (F_SS * W_SS) + (F_ST * W_ST) )
         alpha = SR_TT / SR_ST   (resolution ratio, e.g. 1m/10m = 0.1)
Eq. 12:  W_STSF = Concat( phi(AP(sigmoid(F_SS))), phi(MP(sigmoid(F_SS))),
                          phi(AP(sigmoid(F_ST))), phi(MP(sigmoid(F_ST))) )
         then MLP -> split into W_SS and W_ST
"""
import torch
import torch.nn as nn


class CrossResolutionFusion(nn.Module):
    def __init__(self, channels, alpha=0.1):
        super().__init__()
        self.alpha = alpha                       # Eq. 11 adjustment coefficient
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.shared_conv = nn.Conv2d(channels, channels, 1, bias=False)  # phi
        self.mlp = nn.Sequential(                # compress + integrate (Eq. 12)
            nn.Linear(channels * 4, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels * 2),
        )
        self.channels = channels

    def forward(self, f_ss, f_st):
        B, C = f_ss.shape[0], f_ss.shape[1]
        s_ss, s_st = torch.sigmoid(f_ss), torch.sigmoid(f_st)
        # Eq. 12: shared-conv attention descriptors of both modalities
        desc = torch.cat([
            self.shared_conv(self.ap(s_ss)), self.shared_conv(self.mp(s_ss)),
            self.shared_conv(self.ap(s_st)), self.shared_conv(self.mp(s_st)),
        ], dim=1).view(B, 4 * C)
        w = self.mlp(desc).view(B, 2, C, 1, 1)
        w_ss, w_st = w[:, 0], w[:, 1]            # split into the two modality weights
        # Eq. 11
        return f_ss + self.alpha * ((f_ss * w_ss) + (f_st * w_st))
