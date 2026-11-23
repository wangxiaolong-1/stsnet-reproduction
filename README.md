# STSNet 复现实现

> 论文：*STSNet: A cross-spatial resolution multi-modal remote sensing deep
> fusion network for high resolution land-cover segmentation*，Information
> Fusion 114 (2025) 102689，武汉大学 Yu Beibei / Li Jiayi / Huang Xin。

本项目完整复现了论文提出的 STSNet 网络，用于高分辨率土地覆盖分割。网络融合
两种空间分辨率的遥感数据——1 m 高光谱影像（H²SR，249 波段）与 10 m 多时相
Sentinel-2 影像（31 期 × 10 波段），通过跨分辨率融合实现时空-光谱深度联合解译。

## 网络结构

对应论文 Fig.2，网络由四个模块组成：

| 模块 | 论文章节 | 说明 |
|---|---|---|
| 多尺度空间-光谱编码器 | Sec.2.1, Eq.1-4 | 深度可分离卷积 + 光谱门控；四分支并行，仅首尾两次跨尺度连接 |
| 时空编码器 | Sec.2.2, Eq.5-10 | 光谱卷积 + 3D Patch 划分 + 时空 Transformer 块（ConvLSTM 与 3D 窗口/移位窗口注意力并联）+ Patch Merging |
| 跨分辨率融合模块 | Sec.2.3, Eq.11-12 | 通道注意力 + 分辨率比值系数 α（默认 0.1） |
| 多尺度解码器 | Fig.2 | 1×1 对齐 + 分数卷积上采样 + 逐元素相加 |

## 目录结构

```text
.
├── README.md
└── stsnet_impl/
    ├── README.md          安装依赖与使用说明
    ├── VERIFICATION.md    代码与论文公式核对报告
    ├── requirements.txt   Python 依赖
    ├── stsnet/            模型、数据集和评估指标
    ├── train.py           训练
    ├── eval.py            评估
    ├── data_prep.py       数据准备
    └── test_smoke.py      冒烟测试
```

训练时会生成 `stsnet_impl/checkpoints/`。依赖安装见 [实现说明](stsnet_impl/README.md)。

## 实验配置

| 项 | 值 |
|---|---|
| 优化器 | Adam，lr = 0.001 |
| 损失函数 | 交叉熵 |
| batch size | 8 |
| 训练轮数 | 45（WHU-H2SR-MT） |
| 输入 | H²SR (B,249,300,300) + MT (B,31,10,30,30) |
| 输出 | logits (B,8,300,300) |
| 参数量 | 27.77 M（论文报告 28.71 M） |

## 数据集

WHU-H2SR-MT：沈阳实验区，2531 对样本，8 类地物，6:1:3 划分。托管于 Zenodo
（记录号 10578673，66 个 RAR 分卷、压缩后约 83.5 GB、11 个子区域，解压后更大）。数据准备见 `data_prep.py`。

## 快速开始

```bash
cd stsnet_impl

# 冒烟测试
.\.venv\Scripts\python.exe test_smoke.py

# 合成数据训练（管线验证）
.\.venv\Scripts\python.exe train.py --data synthetic --epochs 8 --batch-size 4

# 全量评估
.\.venv\Scripts\python.exe eval.py --data synthetic --checkpoint checkpoints/stsnet_best.pt

# 真实数据（需先下载约 83.5 GB，Zenodo 需能连通）
.\.venv\Scripts\python.exe data_prep.py download --out data/raw
.\.venv\Scripts\python.exe data_prep.py extract  --root data/raw
.\.venv\Scripts\python.exe data_prep.py probe    --root data/raw/extracted/1
.\.venv\Scripts\python.exe data_prep.py tile     --root data/raw --out data/whu
.\.venv\Scripts\python.exe train.py --data data/whu --epochs 45 --batch-size 8
```

## 进展

- [x] 四大模块实现，冒烟测试（形状/反向传播/α 敏感性）通过
- [x] 合成数据训练验证（8 epoch：loss 2.07 → 1.08，mIoU 2% → 36%）
- [x] 论文尺度前向验证（参数量 27.77 M，与论文 28.71 M 一致）
- [x] 训练 / 评估 / 数据准备脚本齐备
- [x] 逐模块公式核对（Eq.1–18 全部一致，见 VERIFICATION.md）
- [ ] WHU-H2SR-MT 真实数据训练，复现论文表 9/10
- [ ] 迁移至 GPU 服务器完成 45 epoch 完整训练

## 参考

- 论文 DOI: <https://doi.org/10.1016/j.inffus.2024.102689>
- 官方代码: <https://github.com/RS-Mage/STSNet>
- 数据集（Zenodo）: <https://zenodo.org/records/10578673>
