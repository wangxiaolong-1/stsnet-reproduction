"""Training script for STSNet (paper Sec. 4.1 settings).

Paper settings: Adam, lr 0.001, cross-entropy loss, batch 8,
45 epochs on WHU-H2SR-MT (20 on AeroRIT / Sen4AgriNet).

Usage:
    python train.py --data synthetic --epochs 3 --batch-size 2
    python train.py --data data/whu --epochs 45 --batch-size 8
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if False else ".")
from stsnet import STSNet
from stsnet.datasets import SyntheticPairDataset, WHUH2SRMTDataset, WHU_CLASSES
from stsnet.metrics import SegMetrics


def build_model(args):
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
                   help="'synthetic' or path to WHU-H2SR-MT root")
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
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
    p.add_argument("--save", default="checkpoints/stsnet_best.pt")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    if args.data == "synthetic":
        train_set = SyntheticPairDataset(num_samples=64)
        val_set = SyntheticPairDataset(num_samples=16, seed=123)
        names = [f"c{i}" for i in range(args.num_classes)]
    else:
        train_set = WHUH2SRMTDataset(args.data, "train")
        val_set = WHUH2SRMTDataset(args.data, "val")
        args.h2sr_bands, args.mt_bands, args.mt_times = 249, 10, 31
        names = WHU_CLASSES

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=args.workers)

    model = build_model(args).to(device)
    n_params = sum(pp.numel() for pp in model.parameters()) / 1e6
    print(f"model parameters: {n_params:.2f} M | device: {device}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    metrics = SegMetrics(args.num_classes, names)
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss = time.time(), 0.0
        for h2sr, mt, label in train_loader:
            h2sr, mt, label = h2sr.to(device), mt.to(device), label.to(device)
            logits = model(h2sr, mt)
            loss = criterion(logits, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        model.eval()
        metrics.reset()
        with torch.no_grad():
            for h2sr, mt, label in val_loader:
                logits = model(h2sr.to(device), mt.to(device))
                metrics.update(logits.cpu(), label)
        text, res = metrics.summary()
        print(f"epoch {epoch:3d} | loss {total_loss / len(train_loader):.4f} "
              f"| val mIoU {res['miou'] * 100:.2f} | OA {res['oa'] * 100:.2f} "
              f"| {time.time() - t0:.1f}s")
        if res["miou"] > best_miou:
            best_miou = res["miou"]
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
    print(f"best val mIoU: {best_miou * 100:.2f} (saved to {args.save})")


if __name__ == "__main__":
    main()
