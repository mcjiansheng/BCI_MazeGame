# BCIC2a 预处理与三模型实现说明

更新日期：2026-07-22

## 1. 实现范围

项目已完成从 18 个原始 GDF 文件到跨会话模型测试的闭环：

1. 解析 BCIC IV 2a 的 EEG、事件和官方真实标签；
2. 生成宽带数据及 FBCNet 所需九频带输入；
3. 实现 EEG-Conformer、FBCNet、SE-MHAF-Conformer；
4. 对每名受试者独立训练，保存最佳检查点并在独立 E 会话评测；
5. 输出 accuracy、macro-F1、Cohen's κ、混淆矩阵和训练元数据。

核心代码：

- `src/bci_maze/preprocessing.py`
- `src/bci_maze/training.py`
- `src/bci_maze/models/eeg_conformer.py`
- `src/bci_maze/models/fbcnet.py`
- `src/bci_maze/models/se_mhaf_conformer.py`

## 2. 数据集与标签

BCI Competition IV 2a 含 9 名受试者、每名 2 个会话，每个会话 288 次四类运动想象试次。类别为左手、右手、双脚和舌头；记录包含 22 个 EEG 和 3 个 EOG 通道，采样率 250 Hz。本项目只向分类器提供 22 个 EEG 通道。

训练文件 `AxxT.gdf` 和评估文件 `AxxE.gdf` 均使用官方 `classlabel` MAT 文件，内部类别由 1–4 转为 0–3。评估标签来自 [BCI Competition IV 官方结果页](https://bbci.de/competition/iv/results/)，数据定义以 [2a 官方说明](https://www.bbci.de/competition/IV/desc_2a.pdf) 为准。

## 3. 预处理流程

### 3.1 试次提取

- 用 MNE 读取 GDF；
- 找到 trial start 事件 `768`；
- 从每个 trial start 后 2 s 截取到 6 s，共 1000 个采样点；
- 前 22 个通道从 V 转为 μV；
- NaN/Inf 安全替换为有限值；
- 若事件 `1023` 落在该试次范围内，将其记为伪迹试次。

这一时间窗和 4–40 Hz 宽带设置与 [EEG-Conformer 官方 BCIC2a 预处理脚本](https://github.com/eeyhsong/EEG-Conformer/blob/main/preprocessing/BCIIV2a.m) 对齐。

### 3.2 两条信号路径

宽带路径供 EEG-Conformer 和 SE-MHAF-Conformer 使用：

```text
原始试次 → 6 阶 Chebyshev II → 4–40 Hz → 双向零相位滤波 → x
```

滤波器组路径供 FBCNet 使用：

```text
raw_x → [4–8, 8–12, ..., 36–40] Hz 九个子带 → shape=(N,9,22,1000)
```

九个子带按 FBCNet 作者公开的 `filterBank` 设计：通带衰减 3 dB、阻带衰减 30 dB、两侧过渡带 2 Hz，由 `cheb2ord` 自动确定阶数，默认采用因果滤波。设计参考 [FBCNet 官方 transforms.py](https://github.com/ravikiran-mane/FBCNet/blob/master/codes/centralRepo/transforms.py)。

### 3.3 数据泄漏防护

每名受试者独立处理：

1. 去除 T/E 会话中标记为伪迹的试次；
2. T 会话按类别分层划分 80% 训练、20% 验证；
3. 均值和标准差仅用 T 会话的训练子集拟合；
4. 同一统计量应用到验证集和 E 会话；
5. E 会话不参与调参、归一化拟合或早停。

宽带按通道归一化；滤波器组按频带与通道归一化。

## 4. EEG-Conformer

实现参考 [EEG-Conformer 官方仓库](https://github.com/eeyhsong/EEG-Conformer)。输入为 `(B,1,22,1000)`。

| 阶段 | 操作 | 输出尺寸 |
|---|---|---|
| 时域卷积 | Conv2d 1→40，kernel `(1,25)` | `(B,40,22,976)` |
| 空间卷积 | Conv2d 40→40，kernel `(22,1)` | `(B,40,1,976)` |
| 池化 | AvgPool `(1,75)`，stride `(1,15)` | `(B,40,1,61)` |
| Token 化 | 展平空间并转置 | `(B,61,40)` |
| 编码器 | 6 个 Transformer block，10 heads | `(B,61,40)` |
| 分类 | flatten 2440 → 256 → 32 → 4 | `(B,4)` |

每个 Transformer block 使用 Pre-LN、多头自注意力、残差、前馈网络和 dropout。可训练参数量为 789,572。

## 5. FBCNet

实现结构与 [FBCNet 官方 networks.py](https://github.com/ravikiran-mane/FBCNet/blob/master/codes/centralRepo/networks.py) 对齐。输入为 `(B,9,22,1000)`。

| 阶段 | 操作 | 输出尺寸 |
|---|---|---|
| 多视图输入 | 九个 4 Hz 子带 | `(B,9,22,1000)` |
| 空间滤波 | 分组 Conv2d，9 组，每带 32 个滤波器 | `(B,288,1,1000)` |
| 非线性 | BatchNorm + Swish | `(B,288,1,1000)` |
| 时间分段 | 4 个 1 s 区间 | `(B,288,4,250)` |
| 统计聚合 | 每段 `log(clamp(var))` | `(B,288,4)` |
| 分类 | 1152 → 4，最大范数约束 | `(B,4)` |

空间卷积权重最大范数为 2，线性分类器最大范数为 0.5。可训练参数量仅 11,812，是三者中最轻量的模型。训练默认值参照官方 hold-out 脚本的 batch 16、Adam lr 1e-3，并用最长 500 轮、patience 100 作为本项目的可运行上限；官方脚本本身允许 1500 轮、patience 200。

## 6. SE-MHAF-Conformer

这是依据项目已有方案落地的自定义模型，不宣称为外部论文官方代码。输入为 `(B,1,22,1000)`，可训练参数量为 1,076,089。

### 6.1 多尺度时空前端

三个并行分支使用长度 63、31、15 的时域卷积，以覆盖约 252 ms、124 ms、60 ms 的局部模式；每个分支再通过按特征图分组的 `(22,1)` 深度空间卷积，将 22 通道压缩为空间特征。三个分支各输出 32 通道，拼接为 96 通道。

随后使用深度可分离 `(1,15)` 卷积和 `1×1` 点卷积融合为 120 通道，经两次步长 4 的平均池化得到 `(B,120,1,62)`。

### 6.2 SE 通道重标定

对每个通道做全局平均，经过 `120 → 15 → 120` 的两层 MLP 和 Sigmoid 得到权重：

```text
s = mean(x, spatial, time)
w = sigmoid(W2(ReLU(W1(s))))
x_se = x * w
```

### 6.3 MHAF 编码器

62 个时间 token、每个 120 维，进入 6 层、10 头编码器。普通缩放点积注意力为：

```text
S_l = Q_l K_l^T / sqrt(d_head)
```

从第二层起，加入上一层注意力分布的可学习相关项：

```text
S_l' = S_l + sigmoid(alpha_l) * A_(l-1)
A_l  = softmax(S_l')
```

各头输出不只拼接，还通过可学习矩阵 `M` 做头间融合：

```text
H'_i = sum_j softmax(M)[i,j] * H_j
```

最后对 62 个 token 做均值池化，经 `120 → 64 → 4` 分类。该结构的价值需要通过多随机种子、超参数搜索及去 SE/去层间关联/固定头融合矩阵等消融实验进一步验证。

## 7. 训练与评测

| 配置 | EEG-Conformer | FBCNet | SE-MHAF-Conformer |
|---|---:|---:|---:|
| 最大 epochs | 200 | 500 | 200 |
| patience | 40 | 100 | 40 |
| batch size | 64 | 16 | 64 |
| Adam learning rate | 2e-4 | 1e-3 | 2e-4 |
| loss | CrossEntropy | CrossEntropy | CrossEntropy |
| AMP | 开启（CUDA） | 开启（CUDA） | 开启（CUDA） |
| 默认数据增强 | 否 | 否 | 否 |

每名受试者使用 `seed=42+subject`。早停指标为验证准确率；保存验证最优权重后，再计算 E 会话的 accuracy、macro-F1、Cohen's κ 和 4×4 混淆矩阵。

## 8. 扩展接口

- 新模型：在 `src/bci_maze/models/` 新增模块，并在 `models/__init__.py` 的 `build_model` 注册；
- 新预处理：优先扩展 `PreprocessConfig`，将参数写入 `metadata.json`；
- 消融：通过构造函数暴露分支数、SE、MHAF 相关项和头融合开关；
- 在线系统：需额外实现因果滑窗、实时归一化、置信度平滑、模型导出和 LSL/设备接口，当前离线代码不能直接等同于实时部署。

## 9. 已知限制

- 当前正式结果只有一个固定随机种子，尚未报告多次重复实验置信区间；
- 未对三个模型做等预算的超参数搜索；
- EEG-Conformer 官方结果使用更长训练和分段重构增强，本轮默认关闭增强；
- SE-MHAF-Conformer 当前结果低于 FBCNet，说明结构复杂度尚未转化为泛化收益，需要消融和正则化优化；
- 这里只完成 EEG 解码离线基线，尚未与迷宫游戏控制回路集成。

