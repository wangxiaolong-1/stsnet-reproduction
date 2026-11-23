"""Full evaluation + prediction visualization for STSNet.

Computes the paper's Table 9/10 style metrics on any split (train/val/test):
per-class IoU & F1, overall accuracy (OA), mean IoU (mIoU), plus the full
confusion matrix. Optionally dumps per-sample prediction maps (color PNG + raw
npy) for qualitative inspection.

Usage:
    # quick synthetic sanity check
    python eval.py --data synthetic

    # real test set (after preparing data and training)
    python eval.py --data data/whu --split test --checkpoint checkpoints/stsnet_best.pt --save-dir outputs
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from stsnet import STSNet
from stsnet.datasets import SyntheticPairDataset, WHUH2SRMTDataset, WHU_CLASSES
from stsnet.metrics import SegMetrics

# 8-class color palette (R,G,B) used for prediction overlays.
PALETTE = np.array([
    [255, 255,   0],   # 0 paddy      -> yellow
    [205, 133,  63],   # 1 dry farm   -> peru
    [ 34, 139,  34],   # 2 forest     -> forest green
    [ 50, 205,  50],   # 3 grass      -> lime green
    [128,   0,   0],   # 4 building   -> dark red
    [ 64,  64,  64],   # 5 highway    -> dark gray
    [153,  51, 255],   # 6 greenhouse -> purple
    [  0,   0, 255],   # 7 water      -> blue
], dtype=np.uint8)


def render_png(pred, path):
    """Render an (H, W) class map to a palette PNG (needs Pillow)."""
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed, skipping PNG render; saved .npy only)")
        return False
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(PALETTE)):
        rgb[pred == c] = PALETTE[c]
    Image.fromarray(rgb, mode="RGB").save(path)
    return True


def build_model_from_args(args):
    return STSNet(
        num_classes=args.num_classes,
        h2sr_bands=args.h2sr_bands,
        ss_channels=tuple(int(c) for c in args.ss_channels.split(",")),
        mt_bands=args.mt_bands,
        mt_times=args.mt_times,
        st_dims=tuple(int(d) for d in args.st_dims.split(",")),
        st_out_channels=args.st_out,
        alpha=args.alpha,
        ws=tuple(int(w) for w in args.ws.split(",")),
        num_heads=args.heads,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic",
                   help="'synthetic' or path to WHU-H2SR-MT root (with manifest.json)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", default=None, help=".pt checkpoint to load")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--h2sr-bands", type=int, default=16)
    p.add_argument("--mt-bands", type=int, default=4)
    p.add_argument("--mt-times", type=int, default=8)
    p.add_argument("--ss-channels", default="16,24,32,32")
    p.add_argument("--st-dims", default="16,24,32")
    p.add_argument("--st-out", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--ws", default="2,4,4")
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save-dir", default=None, help="dir to dump prediction maps")
    p.add_argument("--max-save", type=int, default=16,
                   help="cap on how many prediction maps to save")
    args = p.parse_args()

    device = torch.device(args.device)

    # ---- dataset -----------------------------------------------------------
    if args.data == "synthetic":
        dataset = SyntheticPairDataset(num_samples=16, seed=123)
        names = [f"c{i}" for i in range(args.num_classes)]
    else:
        dataset = WHUH2SRMTDataset(args.data, args.split)
        args.h2sr_bands, args.mt_bands, args.mt_times = 249, 10, 31
        names = WHU_CLASSES
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    # ---- model -------------------------------------------------------------
    model = build_model_from_args(args).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        # saved args (from train.py) take precedence over CLI defaults
        if "args" in ckpt:
            saved = ckpt["args"]
            args.num_classes = saved.get("num_classes", args.num_classes)
            args.h2sr_bands = saved.get("h2sr_bands", args.h2sr_bands)
            args.mt_bands = saved.get("mt_bands", args.mt_bands)
            args.mt_times = saved.get("mt_times", args.mt_times)
            args.ss_channels = saved.get("ss_channels", args.ss_channels)
            args.st_dims = saved.get("st_dims", args.st_dims)
            args.st_out = saved.get("st_out", args.st_out)
            args.alpha = saved.get("alpha", args.alpha)
            args.ws = saved.get("ws", args.ws)
            args.heads = saved.get("heads", args.heads)
            model = build_model_from_args(args).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"loaded checkpoint {args.checkpoint}")
    n_params = sum(pp.numel() for pp in model.parameters()) / 1e6
    print(f"model parameters: {n_params:.2f} M | device: {device}")

    # ---- evaluate ----------------------------------------------------------
    model.eval()
    metrics = SegMetrics(args.num_classes, names)
    saved = 0
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for h2sr, mt, label in loader:
            logits = model(h2sr.to(device), mt.to(device))
            metrics.update(logits.cpu(), label)
            if save_dir and saved < args.max_save:
                pred = logits.argmax(1).cpu().numpy()
                for i in range(pred.shape[0]):
                    if saved >= args.max_save:
                        break
                    name = f"{args.split}_{saved:04d}"
                    np.save(os.path.join(save_dir, name + ".npy"), pred[i])
                    render_png(pred[i], os.path.join(save_dir, name + ".png"))
                    saved += 1

    text, res = metrics.summary()
    print("\n=== per-class results ===")
    print(text)

    # confusion matrix
    cm = metrics.hist.cpu().numpy()
    header = "".join(f"{n[:4]:>6s}" for n in names)
    print("\n=== confusion matrix (row=true, col=pred) ===")
    print("        " + header)
    for r in range(args.num_classes):
        print(f"{names[r]:<8s}" + "".join(f"{cm[r, c]:6d}" for c in range(args.num_classes)))

    if save_dir:
        np.save(os.path.join(save_dir, "confusion.npy"), cm)
        print(f"\npredictions & confusion saved to {save_dir}")


if __name__ == "__main__":
    main()
