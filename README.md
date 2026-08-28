# BCI Maze：BCIC IV 2a/2b 运动想象解码基线

本仓库实现了 BCI Competition IV 2a/2b 数据预处理、三条基线及多代 SE-MHAF-Conformer 的严格跨会话测试流程：

- EEG-Conformer：时空卷积 + Transformer；
- FBCNet：九频带滤波器组 + 空间卷积 + 分段对数方差；
- SE-MHAF-Conformer：项目自定义的多尺度时域卷积 + SE + 多头注意力融合模型。
- SE-MHAF-Conformer V2：显式 FBC-logvar 主路径 + 轻量稳定 MHAF 残差路径。
- SE-MHAF-Conformer Final：冻结 FBC 主干 + 逐频带 SE + logvar/连续时间 MHAF 专家 + 验证集安全回退。

当前固定种子基准（训练会话建模、评估会话测试、剔除伪迹试次）为：

| 数据集 | EEG-Conformer | FBCNet | SE-MHAF V1 | V2 | Final |
|---|---:|---:|---:|---:|---:|
| BCIC2a：22 通道四分类 | 56.03% | 69.30% | 53.93% | 58.84% | **70.55%** |
| BCIC2b：3 通道二分类 | 76.57% | 76.81% | 75.69% | 80.18% | **80.81%** |

这是本仓库一次可复现实验结果，不等同于论文作者在不同训练轮次、增强策略或重复实验下报告的数值。

## 1. 项目流程

```text
BCIC2a GDF/MAT 或 BCIC2b 官方 BNCI MAT
            │
            ├─ 2a 截取 trial+2–6 s；2b 截取 trial+3–7 s
            ├─ 保留 22 或 3 个 EEG 通道，记录官方伪迹标记
            ├─ EEG-Conformer / SE-MHAF V1：4–40 Hz 宽带数据
            └─ FBCNet / V2 / Final：由原始试次构造九频带
                              │
                              ▼
          T 会话分层划分 80% 训练 / 20% 验证
          E 会话只在最佳验证检查点确定后测试一次
                              │
                              ▼
       accuracy / macro-F1 / Cohen's κ / confusion matrix
```

## 2. 环境安装

推荐 Python 3.10+。PyTorch 应先按本机 CUDA 版本从官方渠道安装，再安装项目：

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install -e .
```

本工作站验证环境为 Python 3.11、PyTorch 2.5.1+cu121、MNE 1.8.0，GPU 为 RTX 4080 Laptop。

## 3. 数据布局与预处理

原始数据和官方标签应为：

```text
data/BCIC2a/
├── A01T.gdf ... A09T.gdf
├── A01E.gdf ... A09E.gdf
└── true_labels/
    └── A01T.mat ... A09E.mat
```

运行：

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_bcic2a.py --overwrite
```

输出写入 `data/processed/bcic2a/`。每个 NPZ 同时保存宽带 `x`、FBCNet 使用的硬件滤波后原始试次 `raw_x`、类别 `y`、伪迹掩码、通道名和采样率。

BCIC2b 从 [BNCI Horizon 2020 官方数据库](https://bnci-horizon-2020.eu/database/data-sets)的 `004-2014` 条目下载后，目录应包含 `B01T/B01E ... B09T/B09E.mat`。运行：

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_bcic2b.py --overwrite
```

## 4. 训练与测试

完整复现实验：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_models.py `
  --dataset bcic2a `
  --models eeg_conformer,fbcnet,se_mhaf_conformer `
  --subjects all `
  --output outputs\bcic2a_benchmark.json
```

BCIC2b 只需将数据集参数改为：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_models.py --dataset bcic2b --subjects all
```

运行优化后的 V2：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_models.py `
  --dataset bcic2b `
  --models se_mhaf_conformer_v2 `
  --subjects all `
  --output outputs\bcic2b_se_mhaf_v2.json
```

运行最终模型：

```powershell
# BCIC2a：logvar 专家
.\.venv\Scripts\python.exe scripts\benchmark_final_model.py `
  --dataset bcic2a --baseline outputs\bcic2a_benchmark.json `
  --subjects all --logvar-only --min-validation-gain 0.05 `
  --output outputs\bcic2a_se_mhaf_final.json

# BCIC2b：logvar + V2 连续时间专家
.\.venv\Scripts\python.exe scripts\benchmark_final_model.py `
  --dataset bcic2b --baseline outputs\bcic2b_benchmark.json `
  --v2 outputs\bcic2b_se_mhaf_v2.json --subjects all `
  --min-validation-gain 0.05 --output outputs\bcic2b_se_mhaf_final.json
```

脚本自动使用模型特定默认值：EEG-Conformer/SE-MHAF-Conformer 为 200 epochs、patience 40、batch 64、lr 2e-4；FBCNet 为 500 epochs、patience 100、batch 16、lr 1e-3。命令行的 `--epochs`、`--patience`、`--batch-size`、`--lr` 可统一覆盖默认值；`--augment` 可为两个 Conformer 启用分段重构增强。

只测试单一模型和受试者：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_models.py --models fbcnet --subjects 1
```

检查点位于 `outputs/checkpoints/<dataset>/<model>/<subject>.pt`。最终指标位于 `outputs/bcic2a_se_mhaf_final.json` 和 `outputs/bcic2b_se_mhaf_final.json`，成本剖析位于 `outputs/model_costs.json`。

## 5. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

当前测试覆盖 22 通道四分类和 3 通道二分类的模型前向/反向传播、参数量差异、九频带处理、BCIC2b MAT 试次提取，以及最终模型零初始化时严格等于 FBCNet 的不变量。

## 6. 目录说明

```text
src/bci_maze/
├── preprocessing.py        # GDF/MAT 读取、宽带与滤波器组处理
├── preprocessing_bcic2b.py # BNCI 004-2014 读取和会话合并
├── training.py             # 数据划分、归一化、训练、早停与评测
└── models/
    ├── eeg_conformer.py
    ├── fbcnet.py
    ├── se_mhaf_conformer.py
    ├── se_mhaf_conformer_v2.py
    ├── se_mhaf_conformer_v3.py
    ├── se_mhaf_conformer_final.py
    └── layers.py
scripts/                    # 可直接运行的预处理和基准脚本
tests/                      # 自动化测试
doc/                        # 原始项目资料、实现说明和实验报告
outputs/                    # 指标 JSON 与模型检查点
```

详细技术设计见 [最终优化与验证报告](doc/SE-MHAF-Conformer_最终优化与验证报告.md)、[BCIC2a 实现说明](doc/BCIC2a预处理与三模型实现说明.md)、[BCIC2b 适配说明](doc/BCIC2b预处理与适配说明.md)和 [V2 优化说明](doc/SE-MHAF-Conformer_V2优化说明.md)，两数据集结果与成本评价见 [综合实验报告](doc/BCIC2a与2b三模型综合实验报告.md)。

## 7. 主要参考

- [BCI Competition IV 2a 官方数据说明](https://www.bbci.de/competition/IV/desc_2a.pdf)
- [BCI Competition IV 2b 官方数据说明](https://www.bbci.de/competition/IV/desc_2b.pdf)
- [BNCI Horizon 2020 数据库](https://bnci-horizon-2020.eu/database/data-sets)
- [EEG-Conformer 官方实现](https://github.com/eeyhsong/EEG-Conformer)
- [FBCNet 官方实现](https://github.com/ravikiran-mane/FBCNet)
- [FBCNet 论文](https://arxiv.org/abs/2104.01233)
