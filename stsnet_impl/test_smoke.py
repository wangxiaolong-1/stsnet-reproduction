"""Smoke tests: shape flow + backward pass for every module and the full model."""
import sys
import torch

from stsnet import (STSNet, MultiScaleSpatialSpectralEncoder,
                    SpatioTemporalEncoder, CrossResolutionFusion,
                    MultiScaleDecoder, ConvLSTM)


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


print("== module tests ==")
# ConvLSTM
x = torch.randn(2, 8, 16, 8, 8)
out = ConvLSTM(16, 32)(x)
check("ConvLSTM (2,8,16,8,8)->(2,8,32,8,8)", out.shape == (2, 8, 32, 8, 8))

# spatial-spectral encoder
enc = MultiScaleSpatialSpectralEncoder(in_bands=16, channels=(16, 24, 32, 32))
feats = enc(torch.randn(2, 16, 64, 64))
check("SS encoder 4 scales /4 /8 /16 /32",
      [f.shape[-1] for f in feats] == [16, 8, 4, 2] and
      [f.shape[1] for f in feats] == [16, 24, 32, 32])

# spatio-temporal encoder
ste = SpatioTemporalEncoder(bands=4, d2=16, d3=24, d4=32, out_channels=32,
                            num_times=8)
out = ste(torch.randn(2, 8, 4, 16, 16))
check("ST encoder (2,8,4,16,16)->(2,32,16,16)", out.shape == (2, 32, 16, 16))

# cross-resolution fusion
fused = CrossResolutionFusion(32, alpha=0.1)(torch.randn(2, 32, 4, 4),
                                             torch.randn(2, 32, 4, 4))
check("fusion keeps (2,32,4,4)", fused.shape == (2, 32, 4, 4))

# decoder
dec = MultiScaleDecoder([16, 24, 32, 32], num_classes=8)
logits = dec(feats)
check("decoder -> full res logits (2,8,64,64)", logits.shape == (2, 8, 64, 64))

print("== full model forward/backward ==")
model = STSNet(num_classes=8, h2sr_bands=16, ss_channels=(16, 24, 32, 32),
               mt_bands=4, mt_times=8, st_dims=(16, 24, 32), st_out_channels=32,
               alpha=0.1, ws=(2, 4, 4), num_heads=2)
h2sr = torch.randn(2, 16, 64, 64)
mt = torch.randn(2, 8, 4, 16, 16)
logits = model(h2sr, mt)
check("STSNet logits (2,8,64,64)", logits.shape == (2, 8, 64, 64))

loss = torch.nn.functional.cross_entropy(
    logits, torch.randint(0, 8, (2, 64, 64)))
loss.backward()
grads = [p.grad is not None and p.grad.abs().sum() > 0
         for p in model.parameters() if p.requires_grad]
check(f"backward: {sum(grads)}/{len(grads)} parameter tensors receive gradient",
      sum(grads) / len(grads) > 0.9)

# alpha sensitivity: changing alpha must change the output
model.zero_grad()
with torch.no_grad():
    model.fusion.alpha = 0.5
    logits2 = model(h2sr, mt)
check("alpha coefficient affects fusion output", not torch.allclose(logits, logits2))

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"\nall smoke tests passed | params (small config): {n_params:.2f} M")
