"""Download, extract and tile the WHU-H2SR-MT dataset (Zenodo record 10578673).

The benchmark is hosted on Zenodo as 66 RAR archives covering 11 sub-regions of
the Shenyang experimental area. This script automates the whole pipeline:

    probe      inspect an extracted region directory and report the *actual*
               internal file format (the released format is not documented, so
               ALWAYS probe first)
    download   fetch all archives from Zenodo (resumable + md5 verified)
    extract    unpack the RAR volumes with 7z / unrar / bsdtar
    tile       crop full-area images + labels into 300 m patches (80 m overlap)
               -> <root>/h2sr/*.npy, <root>/mt/*.npy, <root>/label/*.npy
               + <root>/manifest.json
    manifest   (re)build manifest.json from an existing tile directory

Paper recipe (Sec. 3.2): patches of 300 m x 300 m, 80 m overlap, 2531 pairs,
split 6:1:3. H2SR is 1 m (300 px), Sentinel-2 is 10 m (30 px), 31 dates,
10 bands, 249 spectral bands, 8 classes.

Example:
    .\\.venv\\Scripts\\python.exe data_prep.py probe  --root data/raw/region_1
    .\\.venv\\Scripts\\python.exe data_prep.py download --out data/raw
    .\\.venv\\Scripts\\python.exe data_prep.py extract --root data/raw
    .\\.venv\\Scripts\\python.exe data_prep.py tile --root data/raw --out data/whu
    .\\.venv\\Scripts\\python.exe data_prep.py manifest --root data/whu
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

ZENODO_RECORD = "10578673"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Sentinel-2 bands used by the paper (B2 B3 B4 B5 B6 B7 B8 B8a B11 B12).
# 8 WHU land-cover classes.
WHU_CLASSES = ["paddy", "dry farm", "forest", "grass",
               "building", "highway", "greenhouse", "water"]


# --------------------------------------------------------------------------- #
# Zenodo file list
# --------------------------------------------------------------------------- #
def fetch_zenodo_files():
    """Return [{'name': str, 'size': int, 'md5': str, 'link': str}, ...]."""
    with urllib.request.urlopen(ZENODO_API, timeout=30) as r:
        data = json.load(r)
    out = []
    for f in data.get("files", []):
        out.append({
            "name": f["key"],
            "size": f["size"],
            "md5": f.get("checksum", "").replace("md5:", ""),
            "link": f["links"]["self"],
        })
    return out


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(out_dir, only_prefix=None):
    files = fetch_zenodo_files()
    os.makedirs(out_dir, exist_ok=True)
    total = sum(f["size"] for f in files) / 1e9
    print(f"{len(files)} files, {total:.1f} GB total")
    for i, f in enumerate(files, 1):
        if only_prefix and not f["name"].startswith(only_prefix):
            continue
        dest = os.path.join(out_dir, f["name"])
        if os.path.exists(dest) and _md5(dest) == f["md5"]:
            print(f"[{i}/{len(files)}] skip {f['name']} (complete)")
            continue
        print(f"[{i}/{len(files)}] downloading {f['name']} "
              f"({f['size'] / 1e9:.2f} GB)")
        # resumable download
        req = urllib.request.Request(f["link"])
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        if 0 < existing < f["size"]:
            req.add_header("Range", f"bytes={existing}-")
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab") as w:
            shutil.copyfileobj(r, w)
        if _md5(dest) != f["md5"]:
            print(f"  !! md5 mismatch for {f['name']}, re-download needed")
        else:
            print(f"  ok")


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def _find_extractor():
    for cmd in ("7z", "7za", "unrar", "bsdtar"):
        if shutil.which(cmd):
            return cmd
    return None


def extract(root, prefix=None):
    ex = _find_extractor()
    if ex is None:
        sys.exit("No extractor found. Install 7-Zip ('7z') and add it to PATH.")
    print(f"using extractor: {ex}")
    archives = sorted(f for f in os.listdir(root) if f.lower().endswith(".rar"))
    for a in archives:
        if prefix and not a.startswith(prefix):
            continue
        # each region -> own subdir
        region = a.split("_")[2] if a.startswith(("H2SR_image", "H2SR_label")) else \
                 a.split("_")[1] if a.startswith(("MT_image", "MT_label")) else "0"
        dest = os.path.join(root, "extracted", region)
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(root, a)
        if ex in ("7z", "7za"):
            subprocess.run([ex, "x", "-y", f"-o{dest}", path], check=True)
        elif ex == "unrar":
            subprocess.run([ex, "x", "-y", path, dest + os.sep], check=True)
        elif ex == "bsdtar":
            subprocess.run([ex, "-xf", path, "-C", dest], check=True)
        print(f"extracted {a} -> {dest}")


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def probe(root):
    """Recursively list files and try to identify their raster format."""
    total = 0
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            fp = os.path.join(dirpath, fn)
            size = os.path.getsize(fp)
            total += size
            ext = os.path.splitext(fn)[1].lower()
            note = _guess_format(fp, ext)
            print(f"{size / 1e6:9.1f} MB  {ext or '(none)':<6s} {fp}  {note}")
    print(f"\ntotal {total / 1e9:.2f} GB in {root}")


def _guess_format(fp, ext):
    if ext in (".hdr", ".txt", ".xml", ".json"):
        return "[header/metadata]"
    if ext in (".tif", ".tiff"):
        return "[GeoTIFF]"
    if ext == ".mat":
        return "[MATLAB .mat]"
    if ext in (".img", ".dat", ".raw", ".bin", ".bsq", ".bil", ".bip"):
        return "[ENVI/raw raster, see sibling .hdr]"
    if ext in (".npy",):
        return "[numpy]"
    return "[unknown]"


# --------------------------------------------------------------------------- #
# tile
# --------------------------------------------------------------------------- #
def _read_raster(fp, squeeze=True):
    """Read a raster to a numpy array. Prefers rasterio; falls back to ENVI.

    Returns np.ndarray. Channel-first (C, H, W) for images, (H, W) for labels.
    """
    try:
        import rasterio
        with rasterio.open(fp) as src:
            arr = src.read()
            if squeeze and arr.shape[0] == 1:
                arr = arr[0]
            return arr
    except ImportError:
        pass

    # manual ENVI read (interleaved or BIP/BIL/BSQ per header)
    hdr = fp[:fp.rfind(".")] + ".hdr"
    if not os.path.exists(hdr):
        raise RuntimeError(f"cannot read {fp}: install `rasterio` or provide ENVI .hdr")
    import re
    meta = open(hdr).read()
    samples = int(re.search(r"samples\s*=\s*(\d+)", meta).group(1))
    lines = int(re.search(r"lines\s*=\s*(\d+)", meta).group(1))
    bands = int(re.search(r"bands\s*=\s*(\d+)", meta).group(1))
    interleave = re.search(r"interleave\s*=\s*(\w+)", meta)
    interleave = interleave.group(1).lower() if interleave else "bsq"
    dtype = re.search(r"data type\s*=\s*(\d+)", meta)
    dt = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
          5: np.float64, 12: np.uint16}.get(int(dtype.group(1)), np.float32)
    raw = np.fromfile(fp, dtype=dt)
    if interleave == "bsq":
        arr = raw.reshape(bands, lines, samples)
    elif interleave == "bip":
        arr = raw.reshape(lines, samples, bands).transpose(2, 0, 1)
    else:  # bil
        arr = raw.reshape(lines, bands, samples).transpose(1, 0, 2)
    return arr


def tile(root, out, patch=300, overlap=80, h2sr_res=1.0, mt_res=10.0):
    """Crop full-area rasters into aligned patches and save npy + manifest.

    Expects per-region directories under <root>/extracted/<N>/ containing the
    H2SR image, the MT stack and a label raster (exact filenames discovered
    heuristically below).
    """
    os.makedirs(out, exist_ok=True)
    for sub in ("h2sr", "mt", "label"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    h2sr_px = int(patch / h2sr_res)        # 300
    mt_px = int(patch / mt_res)            # 30
    h2sr_stride = h2sr_px - int(overlap / h2sr_res)   # 220
    mt_stride = mt_px - int(overlap / mt_res)         # 22

    regions = sorted(d for d in os.listdir(os.path.join(root, "extracted"))
                     if os.path.isdir(os.path.join(root, "extracted", d)))
    manifest = []
    idx = 0
    for r in regions:
        base = os.path.join(root, "extracted", r)
        files = [f for f in os.listdir(base)]
        h2sr_f = _pick(files, ["h2sr", "hyperspectral", "image"])
        mt_f = _pick(files, ["mt", "sentinel", "s2", "time"])
        label_f = _pick(files, ["label", "gt", "ground"])
        if not (h2sr_f and mt_f and label_f):
            print(f"  !! region {r}: could not identify files -> {files}")
            continue
        h2sr = _read_raster(os.path.join(base, h2sr_f))      # (C, H, W)
        mt = _read_raster(os.path.join(base, mt_f))          # (T, C, h, w)
        label = _read_raster(os.path.join(base, label_f))    # (H, W)
        print(f"region {r}: h2sr {h2sr.shape}, mt {mt.shape}, label {label.shape}")

        H, W = h2sr.shape[-2:]
        for y in range(0, H - h2sr_px + 1, h2sr_stride):
            for x in range(0, W - h2sr_px + 1, h2sr_stride):
                my, mx = y // (h2sr_res / mt_res), x // (h2sr_res / mt_res)
                h = h2sr[..., y:y + h2sr_px, x:x + h2sr_px]
                m = mt[..., my:my + mt_px, mx:mx + mt_px]
                l = label[y:y + h2sr_px, x:x + h2sr_px]
                if h.shape[-2:] != (h2sr_px, h2sr_px) or l.shape != (h2sr_px, h2sr_px):
                    continue
                name = f"r{r}_{idx:05d}.npy"
                np.save(os.path.join(out, "h2sr", name), h.astype(np.float32))
                np.save(os.path.join(out, "mt", name), m.astype(np.float32))
                np.save(os.path.join(out, "label", name), l.astype(np.uint8))
                manifest.append({"id": name, "region": r, "y": y, "x": x})
                idx += 1

    # deterministic 6:1:3 split, then write manifest
    manifest.sort(key=lambda e: e["id"])
    n = len(manifest)
    n_tr, n_va = int(n * 0.6), int(n * 0.1)
    for i, e in enumerate(manifest):
        e["split"] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {n} patches -> {out}/manifest.json")


def _pick(files, keywords):
    for kw in keywords:
        for f in files:
            if kw in f.lower() and not f.lower().endswith((".hdr", ".xml")):
                return f
    return None


def manifest(out):
    """Rebuild manifest.json from existing h2sr/*.npy tiles (no split change)."""
    import glob
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, "h2sr", "*.npy")))
    entries = []
    for i, name in enumerate(ids):
        n = len(ids)
        split = "train" if i < int(n * 0.6) else "val" if i < int(n * 0.7) else "test"
        entries.append({"id": name, "split": split})
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(entries, f, indent=2)
    print(f"wrote {len(ids)} entries -> {out}/manifest.json")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe"); s.add_argument("--root", required=True)
    s = sub.add_parser("download"); s.add_argument("--out", default="data/raw")
    s.add_argument("--prefix", default=None, help="e.g. MT_ to fetch only MT files")
    s = sub.add_parser("extract"); s.add_argument("--root", required=True)
    s.add_argument("--prefix", default=None)
    s = sub.add_parser("tile"); s.add_argument("--root", required=True)
    s.add_argument("--out", default="data/whu")
    s.add_argument("--patch", type=int, default=300)
    s.add_argument("--overlap", type=int, default=80)
    s = sub.add_parser("manifest"); s.add_argument("--root", required=True)

    args = ap.parse_args()
    if args.cmd == "probe":
        probe(args.root)
    elif args.cmd == "download":
        download(args.out, args.prefix)
    elif args.cmd == "extract":
        extract(args.root, args.prefix)
    elif args.cmd == "tile":
        tile(args.root, args.out, args.patch, args.overlap)
    elif args.cmd == "manifest":
        manifest(args.root)


if __name__ == "__main__":
    main()
