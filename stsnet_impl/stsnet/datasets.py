"""Datasets for STSNet training.

SyntheticPairDataset:
    Random-but-structured fake data for pipeline validation (no download needed).
    A tiny hand-crafted "rule" (band -> class mapping with spatial blobs) gives
    the network something actually learnable, so we can verify loss decreases
    and metrics climb.

WHUH2SRMTDataset (real benchmark):
    Loads the tiled layout produced by `data_prep.py tile`:

        root/h2sr/   <id>.npy   (249, 300, 300) float32   [channel-first]
        root/mt/     <id>.npy   (31, 10, 30, 30) float32  [T, C, h, w]
        root/label/  <id>.npy   (300, 300) uint8, values 0..7
        root/manifest.json      [{"id", "split", ...}, ...]

    The 6:1:3 split is read from manifest.json. If it is missing, the loader
    falls back to the flat h2sr/*.npy listing (same behavior as before).
"""
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset

WHU_CLASSES = ["paddy", "dry farm", "forest", "grass",
               "building", "highway", "greenhouse", "water"]


class SyntheticPairDataset(Dataset):
    """Learnable synthetic multi-modal pairs.

    Class of each 8x8 block is drawn from the band-0 mean of the H2SR block;
    the MT series encodes the same class id in its temporal mean. Labels are
    consistent across modalities so fusion is actually useful.
    """

    def __init__(self, num_samples=64, h2sr_size=64, mt_size=16,
                 h2sr_bands=16, mt_bands=4, mt_times=8, num_classes=8, seed=0):
        self.num_classes = num_classes
        rng = np.random.default_rng(seed)
        self.data = []
        blob = np.outer(np.hanning(h2sr_size), np.hanning(h2sr_size))
        for _ in range(num_samples):
            cls_map = rng.integers(0, num_classes, size=(8, 8))
            label = np.kron(cls_map, np.ones((h2sr_size // 8, h2sr_size // 8),
                                             dtype=np.int64))
            base = np.kron(cls_map.astype(np.float32),
                           np.ones((h2sr_size // 8, h2sr_size // 8)))
            h2sr = (base + blob * 0.5)[None] + rng.normal(0, 0.25,
                    (h2sr_bands, h2sr_size, h2sr_size)).astype(np.float32)
            # MT: class-dependent temporal signature + noise
            t = np.linspace(0, 1, mt_times)
            sig = 0.5 + 0.5 * np.sin(2 * np.pi * (cls_map / num_classes)[..., None] * t)
            sig = np.kron(sig, np.ones((mt_size // 8, mt_size // 8, 1)))  # (h,w,T)
            mt = np.tile(sig[None], (mt_bands, 1, 1, 1)).transpose(3, 0, 1, 2)
            mt = mt.astype(np.float32) + rng.normal(0, 0.15,
                    (mt_times, mt_bands, mt_size, mt_size)).astype(np.float32)
            self.data.append((h2sr, mt, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h2sr, mt, label = self.data[idx]
        return (torch.from_numpy(h2sr).float(),
                torch.from_numpy(mt).float(),
                torch.from_numpy(label))


class WHUH2SRMTDataset(Dataset):
    """Loader for the released WHU-H2SR-MT benchmark (tiled npy layout)."""

    def __init__(self, root, split="train", train_ratio=0.6, val_ratio=0.1):
        self.root = root
        # Prefer the manifest written by data_prep.py (deterministic 6:1:3).
        manifest_path = os.path.join(root, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                entries = json.load(f)
            self.files = [e["id"] for e in entries if e.get("split", "train") == split]
        else:
            self.files = sorted(
                os.path.basename(p) for p in glob.glob(os.path.join(root, "h2sr", "*.npy")))
            n = len(self.files)
            n_train, n_val = int(n * train_ratio), int(n * val_ratio)
            if split == "train":
                self.files = self.files[:n_train]
            elif split == "val":
                self.files = self.files[n_train:n_train + n_val]
            else:
                self.files = self.files[n_train + n_val:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        # H2SR: (249, 300, 300) channel-first (as written by data_prep tile)
        h2sr = np.load(os.path.join(self.root, "h2sr", name))
        # MT: (31, 10, 30, 30) [T, C, h, w]
        mt = np.load(os.path.join(self.root, "mt", name))
        label = np.load(os.path.join(self.root, "label", name))     # (300, 300)

        # Normalise array axes defensively: WHU-H2SR has 249 bands, MT has 10.
        # Channel-first is (249, H, W) / (T, 10, h, w); channel-last is
        # (H, W, 249) / (T, h, w, 10).
        if h2sr.ndim == 3 and h2sr.shape[-1] == 249 and h2sr.shape[0] != 249:
            h2sr = h2sr.transpose(2, 0, 1)              # (H, W, C) -> (C, H, W)
        if mt.ndim == 4 and mt.shape[-1] == 10 and mt.shape[1] != 10:
            mt = mt.transpose(0, 3, 1, 2)               # (T, h, w, C) -> (T, C, h, w)

        h2sr = torch.from_numpy(h2sr).float() / 10000.0
        mt = torch.from_numpy(mt).float() / 10000.0
        return h2sr, mt, torch.from_numpy(label).long()
