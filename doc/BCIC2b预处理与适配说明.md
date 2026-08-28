# BCIC2b 预处理与三模型适配说明

更新日期：2026-07-22

## 1. 数据来源与许可

数据从 [BNCI Horizon 2020 官方数据库](https://bnci-horizon-2020.eu/database/data-sets)的 `004-2014` 条目下载。该条目是 BCI Competition IV 2b 的官方整理版，许可为 CC BY-ND 4.0。实际文件由页面重定向到格拉茨工业大学服务器。

共下载 18 个文件：`B01T/B01E ... B09T/B09E.mat`。每个源文件的 SHA-256 已写入 `data/processed/bcic2b/metadata.json`，可用于完整性复核。

官方数据说明见 [BCI Competition IV 2b description](https://www.bbci.de/competition/iv/desc_2b.pdf)。数据包含 9 名受试者、3 个双极 EEG 通道 C3/Cz/C4、3 个 EOG 通道、250 Hz 采样和左右手两类运动想象。EOG 不进入分类器。

## 2. 会话划分

BNCI 文件保持了原始五会话结构：

- `BxxT.mat`：01T、02T、03T 三个会话，作为训练开发数据；
- `BxxE.mat`：04E、05E 两个会话，作为独立评估数据。

不同受试者的 T 文件含 400–440 个试次，E 文件含 280–320 个试次。左右手原始标签严格平衡。官方专家伪迹标记被保留，正式实验只统计 clean trials。

## 3. 试次提取

BNCI MAT 中每个 session 包含：

- `X`：连续信号，前 3 列 EEG，后 3 列 EOG；
- `trial`：MATLAB 一基 trial start 位置；
- `y`：1=左手、2=右手；
- `artifacts`：逐试次伪迹标记；
- `fs`：250 Hz。

处理步骤：

1. 将 trial 位置转换为 Python 零基索引；
2. 截取 trial start 后 3–7 秒，共 1000 点；
3. 只保留 C3/Cz/C4；BNCI MAT 的幅值已经是 μV；
4. 保存未加软件宽带滤波的 `raw_x`；
5. 用 6 阶 Chebyshev II、60 dB 阻带衰减、双向零相位滤波生成 4–40 Hz 的 `x`；
6. FBCNet 从 `raw_x` 生成 4–8 至 36–40 Hz 的九频带输入。

输出文件为 `data/processed/bcic2b/B01T.npz ... B09E.npz`，字段包括 `x/raw_x/y/artifact/session_id/channel_names/sfreq`。

## 4. 模型适配

三个模型均通过构造参数动态适配：

```python
build_model(model_name, n_channels=3, n_times=1000, n_classes=2)
```

- EEG-Conformer：空间卷积核从 `(22,1)` 改为 `(3,1)`，分类输出由 4 改为 2；
- FBCNet：每频带空间卷积覆盖 3 通道，分类输出改为 2；
- SE-MHAF-Conformer：三个分支的深度空间卷积核改为 `(3,1)`，分类输出改为 2；
- 训练代码从数据动态推断通道数、时间点数和类别数；
- 混淆矩阵及分段重构增强不再硬编码四类。

## 5. 评测协议

1. T 文件中的三个会话合并；
2. 排除 T/E 中的伪迹试次；
3. T clean trials 按类别分层划分 80% 训练、20% 验证；
4. 归一化统计量只在训练子集拟合；
5. 验证准确率用于早停和检查点选择；
6. E 文件的两个会话只在最佳检查点确定后评测；
7. 指标为 accuracy、macro-F1、Cohen's κ、混淆矩阵。

这是一种 trial-level、跨会话离线分类协议；它不是原比赛要求的逐采样连续因果输出协议。因此，本项目结果用于三模型统一工程比较，不应直接与原比赛排行榜数值等同。

## 6. 复现命令

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_bcic2b.py --overwrite

.\.venv\Scripts\python.exe scripts\benchmark_models.py `
  --dataset bcic2b `
  --models eeg_conformer,fbcnet,se_mhaf_conformer `
  --subjects all `
  --output outputs\bcic2b_benchmark.json
```

