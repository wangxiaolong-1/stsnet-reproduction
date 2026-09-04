# STSNet 复现实现

论文：*STSNet: A cross-spatial resolution multi-modal remote sensing deep fusion
network for high resolution land-cover segmentation* (Information Fusion 114,
2025, 102689)

官方仓库（仅含 2 个文件，时空编码器与训练代码缺失）：`../official_STSNet`
本实现：完整四大模块 + 训练管线，纯 PyTorch，无其他依赖。

## 文件与论文对应关系

| 文件 | 论文章节 | 内容 |
|---|---|---|
| `stsnet/spatial_spectral_encoder.py` | 2.1, Eq.1-4 | DSC、光谱门控、四分支、两次跨尺度连接 |
| `stsnet/spatio_temporal_encoder.py` | 2.2, Eq.5-10 | 光谱卷积、3D patch 划分、时空 Transformer 块（ConvLSTM ∥ 3D W/SW-MSA）、Patch Merging |
| `stsnet/cross_resolution_fusion.py` | 2.3, Eq.11-12 | 通道注意力 + 分辨率比值系数 α |
| `stsnet/decoder.py` | Fig.2-④ | 1×1 对齐 + 分数卷积（转置卷积）上采样 + 逐元素相加 |
| `stsnet/model.py` | Fig.2 | STSNet 整网组装 |
| `stsnet/metrics.py` | Eq.13-18 | IoU / F1 / OA / mIoU |
| `stsnet/datasets.py` | Sec.3.2 | 合成数据（管线验证）+ WHU-H2SR-MT 加载器（读 manifest.json 的 6:1:3 划分） |
| `train.py` | Sec.4.1 | Adam lr=0.001, CE loss, batch 8, 45 epochs |
| `eval.py` | Sec.4.2 / Table 9-10 | 全量评估：逐类 IoU/F1、OA、mIoU、混淆矩阵 + 预测图保存 |
| `data_prep.py` | Sec.3.2 | WHU-H2SR-MT 下载/解压/探查格式/切块 → manifest.json |

## 使用

> 依赖已装到项目自带虚拟环境 `.venv`（torch 2.14.0+cpu 等）。
> 直接用 `.venv\Scripts\python.exe` 运行即可，无需系统安装 torch。

```bash
# （可选）激活虚拟环境后，即可直接 `python ...`
.\.venv\Scripts\Activate.ps1

# 冒烟测试（形状 + 反向传播 + α 敏感性）
.\.venv\Scripts\python.exe test_smoke.py

# 合成数据训练（验证管线，CPU 可跑）
.\.venv\Scripts\python.exe train.py --data synthetic --epochs 8 --batch-size 4

# 全量评估（逐类 IoU/F1、OA/mIoU、混淆矩阵 + 预测图）
.\.venv\Scripts\python.exe eval.py --data synthetic --batch-size 4 \
    --checkpoint checkpoints/stsnet_best.pt --save-dir outputs

# 真实数据：先准备数据（下载→解压→探查格式→切块→manifest）
.\.venv\Scripts\python.exe data_prep.py download --out data/raw
.\.venv\Scripts\python.exe data_prep.py extract  --root data/raw
.\.venv\Scripts\python.exe data_prep.py probe    --root data/raw/extracted/1
.\.venv\Scripts\python.exe data_prep.py tile     --root data/raw --out data/whu
.\.venv\Scripts\python.exe train.py --data data/whu --epochs 45 --batch-size 8
.\.venv\Scripts\python.exe eval.py  --data data/whu --split test \
    --checkpoint checkpoints/stsnet_best.pt --save-dir outputs
```

> 若因 PowerShell 执行策略无法激活，用 `.\.venv\Scripts\python.exe 脚本名` 直接运行即可。
> 依赖清单见 `requirements.txt`。

论文尺度配置（默认参数即可）：H2SR `(B,249,300,300)` + MT `(B,31,10,30,30)`
→ logits `(B,8,300,300)`，参数量 27.77M（论文报告 28.71M）。

## 真实数据集 WHU-H2SR-MT

托管在 **Zenodo**（记录号 `10578673`，DOI 10.5281/zenodo.10578673），共 **66 个
RAR 文件、约 145 GB**，覆盖沈阳实验区 **11 个子区域**，每区四件套：

| 文件 | 含义 |
|---|---|
| `H2SR_image_<N>[_<卷>].rar` | 高光谱影像（249 波段，1 m，体积大故分卷） |
| `H2SR_label_<N>.rar` | 地表覆盖标签 |
| `MT_image_<N>.rar` | Sentinel-2 多时相（31 期 × 10 波段，10 m） |
| `MT_label_<N>.rar` | 标签（同 H2SR label） |

论文切块口径（Sec.3.2）：300 m × 300 m、80 m 重叠 → 2531 对样本，6:1:3 划分。

> ⚠️ 官方**未在记录里说明解压后的内部文件格式/命名**（可能是 ENVI `.img/.hdr`、
> GeoTIFF `.tif` 或 MATLAB `.mat`）。因此 `data_prep.py` 的流程是：
> **先 `probe` 探查真实格式，再据此微调 `tile` 的读取逻辑**（`_read_raster` 已
> 内置 rasterio 优先、ENVI 手解兜底）。不要跳过 probe 直接切块。

切块后目录即 `data/whu`：`h2sr/*.npy` `(249,300,300)`、`mt/*.npy`
`(31,10,30,30)`、`label/*.npy` `(300,300)`，加 `manifest.json` 记录 6:1:3 划分，
`datasets.py` 据此加载。

## 与论文/官方代码的已知偏差

1. 论文 patch partition 的张量记号有歧义，这里按 Swin 惯例把 2×2×2 体素折叠进通道维；
2. 官方 `fusion.py` 的融合实现为 `x1 + 0.01 * out`（无显式 α），本文按论文公式 11
   实现为可配置 α（默认 0.1 = 1m/10m）；
3. 时空编码器官方未放出，结构按论文 2.2 节文字描述实现（3DSwinT 骨架 +
   ConvLSTM/3D 注意力并联）；
4. 窗口注意力采用 padding-分区-去 padding 的简化实现（未做 Swin 的注意力掩码），
   在特征尺寸非窗口整数倍时行为等价性略有差异。

## 环境隔离

本机为 CPU 验证环境，依赖装在项目虚拟环境 `.venv`（torch 2.14.0+cpu）。
到服务器上换 CUDA 版 torch 即可训练：
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
