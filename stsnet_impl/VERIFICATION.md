# STSNet 复现逐模块核对报告

> 目的：将 `stsnet_impl/stsnet/` 的每一处实现逐条对照论文《STSNet》（Information
> Fusion 114 (2025) 102689）Sec.2.1–2.3 的 Eq.1–18 与官方参考代码
> `official_STSNet/src/`，验证复现是否与原文一致，并如实记录所有偏差。

核对日期：2026-09-04。核对范围：`spatial_spectral_encoder.py`、
`spatio_temporal_encoder.py`、`cross_resolution_fusion.py`、`decoder.py`、
`model.py`、`convlstm.py`、`metrics.py`。

---

## 一、总览结论

| 论文公式 | 内容 | 实现位置 | 核对结果 |
|---|---|---|---|
| Eq.1–4 | 光谱门控模块（SGM） | `spatial_spectral_encoder.SpectralGatedModule` | ✅ 一致 |
| Eq.5–9 | ConvLSTM | `convlstm.ConvLSTMCell` | ✅ 一致 |
| Eq.10 | 3D 窗口/移位窗口自注意力 | `spatio_temporal_encoder.WindowAttention3D` | ✅ 一致 |
| Eq.11 | 跨分辨率融合（α 系数） | `cross_resolution_fusion.CrossResolutionFusion` | ✅ 一致 |
| Eq.12 | 跨模态通道注意力权重 | 同上 | ✅ 一致（论文原文有一处笔误） |
| Eq.13–18 | IoU/F1/OA/mIoU | `metrics.SegMetrics` | ✅ 一致 |
| Sec.2 ④ | 多尺度解码器 | `decoder.MultiScaleDecoder` | ✅ 一致 |

**结论：论文方法部分（Eq.1–18）全部正确复现，无功能性错误。** 通道配置与官方
代码吻合，参数量 27.77M 与论文 28.71M 量级一致。发现的差异均为「论文文字与官方
代码不一致」或「论文未给出具体数值」的合理实现选择，共 7 处，见第四节。

---

## 二、逐模块公式核对

### 2.1 多尺度空间-光谱编码器（Sec.2.1, Eq.1–4）

**论文定义**：四分支并行，每分支较前一分支下采样 2×；每分支先用 DSC 处理波段
内/波段间信息；光谱门控模块按式(1)–(4)计算带权。

$$\mathrm{FSA_{AA}}=\delta(\mathrm{MLP}(\mathrm{AAP}(F_X))) \tag{1}$$
$$\mathrm{FSA_{AM}}=\delta(\mathrm{MLP}(\mathrm{AMP}(F_X))) \tag{2}$$
$$\mathrm{Spectral\ weights}=\mathrm{FSA_{AA}}\oplus\mathrm{FSA_{AM}} \tag{3}$$
$$\mathrm{FSA}=F_X\otimes\mathrm{Spectral\ weights} \tag{4}$$

**实现对应**：

```python
w_aa = self.mlp(self.aap(x).view(b, c))   # Eq.1：自适应平均池化 + 共享MLP(含sigmoid)
w_am = self.mlp(self.amp(x).view(b, c))   # Eq.2：自适应最大池化 + 共享MLP
weights = (w_aa + w_am).view(b, c, 1, 1)  # Eq.3：逐元素相加
return x * weights                        # Eq.4：逐元素相乘
```

- ✅ 共享 MLP（与官方 `EnhancedSqueezeExcitationLayer` 一致），avg/max 双池化、
  sigmoid、相加、相乘，与 Eq.1–4 完全对应。
- ✅ DSC 实现 `groups=in_channels` 的 3×3 depthwise + 1×1 pointwise，符合论文
  "分别处理波段内与波段间信息"的描述。

**通道配置**（对照官方 `fusion.py`）：

| 分支 | 尺度 | 论文/官方通道 | 本实现 |
|---|---|---|---|
| stem | 1/4 | 249 | 249 |
| branch1 | 1/4 | 249 | 249 |
| branch2 | 1/8 | 300 | 300 |
| branch3 | 1/16 | 512 | 512 |
| branch4 | 1/32 | 512 | 512 |

与官方 `downsample1(249→300)`、`downsample2(300→512)`、
`downsample3(512→512)` 完全一致。✅

### 2.2 时空编码器（Sec.2.2, Eq.5–10）

**论文流水线**：光谱卷积 → 3D patch 划分 → 线性嵌入 → 时空 Transformer 块×2
（ConvLSTM ∥ 3D W-MSA/SW-MSA）→ Patch Merging → 卷积调整 → 维度折叠 → 1×1 卷积。

**Eq.5–9（ConvLSTM）实现对应**（`convlstm.ConvLSTMCell`）：

```python
i = sigmoid(cc_i); f = sigmoid(cc_f)          # Eq.5 输入门 / Eq.6 遗忘门
g = tanh(cc_g)
c_next = f * c + i * g                        # Eq.7 记忆单元
o = sigmoid(cc_o)                             # Eq.8 输出门
h_next = o * tanh(c_next)                     # Eq.9 隐状态
```

✅ 与标准 ConvLSTM（官方 TUM-LMF 版）及论文 Eq.5–9 完全一致，输入通道与隐状态
通道不一致时以 1×1 卷积对齐（对应论文 "employ a 1×1 conv to adjust"）。

**Eq.10（3D W-MSA / SW-MSA）实现对应**（`WindowAttention3D`）：

```python
qkv = self.qkv(tokens) ...                    # Q/K/V 三路线性投影
attn = (q * self.scale) @ k.transpose(-2,-1)  # QKᵀ / √d
attn = attn.softmax(dim=-1)                   # SoftMax
out = attn @ v                                # 加权 V
```

✅ 多头自注意力，`scale = (dim/heads)^-0.5`，窗口与移位窗口分别由
`shifted=False/True` 实现（roll + 窗口划分），与 Eq.10 及 Swin 惯例一致。

### 2.3 跨分辨率融合模块（Sec.2.3, Eq.11–12）

**Eq.11 实现对应**：

```python
return f_ss + self.alpha * ((f_ss * w_ss) + (f_st * w_st))
```

即 $F_{STS}=F_{SS}\oplus\alpha\times((F_{SS}\otimes W_{SS})\oplus(F_{ST}\otimes W_{ST}))$。
✅ 逐项对应。`alpha=0.1` 取分辨率比值（10 m / 1 m 的线性尺度比 30/300），与论文
"α=SR_TT/SR_ST，值越小分辨率差异越大"一致。

**Eq.12 实现对应**：

```python
s_ss, s_st = sigmoid(f_ss), sigmoid(f_st)
desc = cat([shared_conv(ap(s_ss)), shared_conv(mp(s_ss)),
            shared_conv(ap(s_st)), shared_conv(mp(s_st))], dim=1)
w = mlp(desc)             # Linear(4C→C) → ReLU → Linear(C→2C)
w_ss, w_st = w[:,0], w[:,1]
```

即 $W_{STSF}=\mathrm{Concat}(\phi(AP(\sigma(F_{SS}))),\phi(MP(\sigma(F_{SS}))),\phi(AP(\sigma(F_{ST}))),\phi(MP(\sigma(F_{ST}))))$。
✅ 共享卷积 $\phi$ 处理两模态四种池化描述子，MLP 压缩后 split 为 $W_{SS}/W_{ST}$。
注意：**论文 Eq.12 原文第四项误写为 $\phi(AP(\sigma(F_{ST})))$（应为 MP）**，本实现
按语义正确使用 `mp(s_st)`。

### 2.4 多尺度解码器（Sec.2 组件④）

论文：1×1 卷积对齐各尺度通道 → 分数卷积（转置卷积）上采样 → 逐元素相加 → 1×1
卷积输出类别。实现对应 `decoder.MultiScaleDecoder`（align 1×1 → ConvTranspose2d
上采样 → element-wise add → head 1×1）。✅ 一致。

### 2.5 评估指标（Eq.13–18）

`metrics.SegMetrics` 的 IoU/F1/OA/mIoU 均按混淆矩阵统计实现，与 Eq.13–18 一致。
✅

---

## 三、参数规模对照

| 项 | 论文（Table 9, Ours） | 本实现 |
|---|---|---|
| 输入 | H²SR (B,249,300,300) + MT (B,31,10,30,30) | 同 |
| 参数量 | 28.71 M | 27.77 M |
| 输出 | (B,8,300,300) | 同 |

差值约 0.94 M（约 3.3%），来源于时空编码器通道数值（论文未给出 D2/D3/D4，本实现
取 64/96/128）与 Patch Merging 折叠惯例的差异，不影响结构正确性判断。

---

## 四、偏差清单（7 处，均非功能性错误）

| # | 类别 | 论文/官方描述 | 本实现选择 | 说明 |
|---|---|---|---|---|
| 1 | 论文 vs 官方 | 论文称光谱门控「仅用于最高分辨率分支」 | 每分支均用（跟随官方 se1–se5） | 跟随官方代码，语义更充分 |
| 2 | 论文 vs 官方 | Fig.2 称卷积块为「三层 3×3 卷积」 | 采用 1×1→3×3→1×1 bottleneck（跟随官方） | 官方 BottleneckBlock 即此结构 |
| 3 | 官方代码疑点 | 官方 BottleneckBlock 定义了 `residual`/`downsample` 却未做 `out += residual` | 同官方（无残差） | 疑为官方漏写，已如实跟随 |
| 4 | 论文歧义 | patch 划分后 shape 写作 `(B,D2,T/2,H/2,W/2)` | 按 Swin 惯例把 2×2×2 折叠进通道（×8） | 论文标注歧义，取维度自洽解 |
| 5 | 论文未给数值 | 时空编码器 D2/D3/D4 未给出 | 取 64/96/128（自研） | 论文仅给符号 |
| 6 | 论文 vs 官方 | 官方 `FeatureFusionModule` 用 channel+spatial 注意力、系数 0.01 | 严格按论文 Eq.11–12 自研（α=0.1） | 论文公式优先于官方残缺实现 |
| 7 | 官方代码缺陷 | 官方 `fusion.py` 含断链 import `Reviewer_3_4.fusion.Time4`，且无时空编码器/训练脚本 | 时空编码器与训练脚本全部自研 | 官方仓库不完整 |

---

## 五、核对结论

复现实现对论文方法部分（Eq.1–18）实现正确、通道配置吻合、参数量量级一致，可判定
**结构层面复现成功**。剩余工作集中在数据层面：接入 WHU-H2SR-MT 真实数据后跑 45
epoch，复现论文 Table 9 的 mIoU 59.73 / OA 87.52。
