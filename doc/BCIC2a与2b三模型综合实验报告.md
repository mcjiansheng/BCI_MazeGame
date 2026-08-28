# BCIC2a 与 BCIC2b 模型综合实验报告

更新日期：2026-07-22

## 摘要

本报告在统一代码框架下比较 EEG-Conformer、FBCNet、SE-MHAF-Conformer V1/V2 及受保护双专家最终版。BCIC2a 采用 22 通道四分类，BCIC2b 采用 3 通道左右手二分类；两者都使用受试者内、跨会话测试，排除官方伪迹试次，并且测试会话不参与归一化拟合、模型选择或早停。

主要结论：

- 最终模型在 BCIC2a 达到 70.55%，比 FBCNet 高 1.25 个百分点；
- 最终模型在 BCIC2b 达到 80.81%，比 FBCNet/V2 高 4.00/0.63 个百分点；
- FBCNet 的参数、检查点和 GPU 网络推理成本最低，但九频带 CPU 预处理增加约 4.4 ms/试次；
- 若包含信号处理，EEG-Conformer 当前单试次端到端延迟略低；若批处理或数据已预先生成频带，FBCNet 吞吐量最高；
- 最终模型通过独立训练并冻结 FBC 主干、零初始化残差、5% 验证增益门槛和专家路由，在两套数据集上均超过原始基线；代价是九频带预处理和额外 MHAF 推理。

## 1. 数据集和实验设计

| 属性 | BCIC2a | BCIC2b |
|---|---:|---:|
| 受试者 | 9 | 9 |
| EEG 通道 | 22 | 3（C3/Cz/C4） |
| 类别 | 4：左/右手、双脚、舌头 | 2：左/右手 |
| 采样率 | 250 Hz | 250 Hz |
| 分析窗 | trial+2–6 s | trial+3–7 s |
| 训练源 | T 会话 | 01T–03T |
| 独立测试源 | E 会话 | 04E–05E |
| 测试 clean trials/人 | 215–283 | 228–307 |

BCIC2b 官方说明指出其包含 9 人、每人 5 会话，前三个会话有标签、后两个用于评估；3 个 EEG 通道为双极 C3/Cz/C4，采样率 250 Hz。参见 [BCIC2b 官方说明](https://www.bbci.de/competition/iv/desc_2b.pdf)和 [BNCI 004-2014 官方条目](https://bnci-horizon-2020.eu/database/data-sets)。

两套实验均使用：分层 80/20 训练验证划分、固定种子 `42+subject`、Adam、交叉熵、验证准确率早停、CUDA AMP。原始三模型与 V2 基准未启用数据增强；最终版冻结 FBC 主干后，残差专家使用论文规定的分段重构增强。

## 2. 总体准确率

| 数据集 | 模型 | Accuracy（均值 ± 受试者总体标准差） | Macro-F1 | Cohen's κ |
|---|---|---:|---:|---:|
| BCIC2a | FBCNet | **69.30% ± 14.19%** | **68.28%** | **0.5910** |
| BCIC2a | EEG-Conformer | 56.03% ± 16.68% | 52.10% | 0.4135 |
| BCIC2a | SE-MHAF-Conformer V1 | 53.93% ± 14.25% | 52.44% | 0.3860 |
| BCIC2a | SE-MHAF-Conformer V2 | 58.84% ± 14.82% | 57.30% | 0.4516 |
| BCIC2a | SE-MHAF-Conformer Final | **70.55% ± 14.23%** | **69.70%** | **0.6075** |
| BCIC2b | FBCNet | 76.81% ± 13.83% | 76.25% | 0.5358 |
| BCIC2b | EEG-Conformer | 76.57% ± 14.86% | 75.43% | 0.5367 |
| BCIC2b | SE-MHAF-Conformer V1 | 75.69% ± 14.21% | 74.97% | 0.5192 |
| BCIC2b | SE-MHAF-Conformer V2 | 80.18% ± 10.08% | 80.04% | 0.6044 |
| BCIC2b | SE-MHAF-Conformer Final | **80.81% ± 10.74%** | **80.51%** | **0.6144** |

最终版相对 V1 的准确率变化为：BCIC2a `+16.62`、BCIC2b `+5.12` 个百分点。BCIC2b 相对 BCIC2a 的高准确率主要反映二分类更简单、训练试次更多以及数据结构不同，不能解释为同一任务上的算法改进。

## 3. BCIC2b 逐受试者结果

| Subject | Clean test trials | EEG-Conformer | FBCNet | SE-MHAF V1 | V2 | Final |
|---:|---:|---:|---:|---:|---:|---:|
| B01 | 228 | 68.42% | 67.11% | 68.86% | 74.56% | **76.75%** |
| B02 | 245 | 54.29% | 49.80% | 57.96% | **66.12%** | 64.49% |
| B03 | 230 | 53.04% | 63.91% | 48.26% | **64.35%** | 63.91% |
| B04 | 307 | 96.42% | **97.39%** | 93.16% | 97.39% | 97.39% |
| B05 | 273 | 73.26% | 90.11% | 73.99% | **90.84%** | 90.11% |
| B06 | 251 | **82.87%** | 77.29% | 82.07% | 79.68% | 77.29% |
| B07 | 232 | 78.88% | 76.72% | 81.47% | 84.05% | **88.36%** |
| B08 | 230 | **92.61%** | 86.96% | 91.74% | 82.61% | 86.96% |
| B09 | 245 | **89.39%** | 82.04% | 83.67% | 82.04% | 82.04% |

最终版在 B01、B07 上进一步超过 V2；B02/B03/B05/B06 则由 V2 或其他模型更高。最终版的优势来自受试者级验证路由和安全回退，而不是每名受试者都强制启用同一残差。

## 4. 模型规模与显存

| 数据集 | 模型 | 参数量 | 平均检查点 | 训练步峰值显存/实际峰值 |
|---|---|---:|---:|---:|
| BCIC2a | EEG-Conformer | 789,572 | 3.034 MB | 370.8 MB（合成训练步） |
| BCIC2a | FBCNet | **11,812** | **0.051 MB** | **88.0 MB（合成训练步）** |
| BCIC2a | SE-MHAF-Conformer V1 | 1,076,089 | 4.141 MB | 631.3 MB（合成训练步） |
| BCIC2a | SE-MHAF-Conformer V2 | 78,142 | 0.327 MB | 约 137.3 MB（正式训练） |
| BCIC2a | SE-MHAF-Conformer Final（logvar） | 416,154 | 1.614 MB | 约 336.3 MB（正式训练均值） |
| BCIC2b | EEG-Conformer | 759,106 | 2.918 MB | 约 206.6 MB（正式训练） |
| BCIC2b | FBCNet | **4,034** | **0.021 MB** | **约 72.3 MB（正式训练）** |
| BCIC2b | SE-MHAF-Conformer V1 | 1,074,135 | 4.133 MB | 约 408.1 MB（正式训练） |
| BCIC2b | SE-MHAF-Conformer V2 | 69,352 | 0.294 MB | 约 89.6 MB（正式训练） |
| BCIC2b | SE-MHAF-Conformer Final（双专家） | 473,434 | 1.858 MB | 约 384.0 MB（正式训练均值） |

FBCNet 的规模优势仍然明显。最终版比 V1 小约 56%–61%，但为获得更高准确率保留 FBC 主干和一个或两个 MHAF 专家，因此规模、检查点和显存高于 V2。

## 5. 正式训练开销

以下时间在 RTX 4080 Laptop GPU 上测得，包含训练循环、每轮验证和最终测试，不包含离线文件读取与滤波器组生成。模型使用不同早停策略，因此必须同时看总时间和每 epoch 时间。

| 数据集 | 模型 | 平均耗时/受试者 | 平均训练轮数 | 加权耗时/epoch |
|---|---|---:|---:|---:|
| BCIC2a | EEG-Conformer | 16.97 s | 119.8 | 0.142 s |
| BCIC2a | FBCNet | 80.74 s | 286.6 | 0.282 s |
| BCIC2a | SE-MHAF-Conformer V1 | 27.81 s | 112.2 | 0.248 s |
| BCIC2a | SE-MHAF-Conformer V2 | 48.87 s | 175.3 | 0.279 s |
| BCIC2a | SE-MHAF-Conformer Final | 34.57 s | 127.2 | 0.272 s |
| BCIC2b | EEG-Conformer | **5.90 s** | 75.4 | 0.078 s |
| BCIC2b | FBCNet | 7.48 s | 152.8 | **0.049 s** |
| BCIC2b | SE-MHAF-Conformer V1 | 7.94 s | 74.4 | 0.107 s |
| BCIC2b | SE-MHAF-Conformer V2 | 34.89 s | 143.0 | 0.244 s |
| BCIC2b | SE-MHAF-Conformer Final | 18.98 s | 101.3 | 0.187 s |

BCIC2a 的 FBCNet 总时间最高并不代表其单次网络计算最重：它使用 batch 16、最多 500 轮、patience 100，每个 epoch 的优化步数和实际轮数都更多。最终版复用已经训练完成的 FBC 检查点，只训练残差专家，因此这里的 34.57/18.98 s 不包含一次性的 FBC 或 V2 预训练时间。

## 6. 推理和信号处理开销

测量环境：RTX 4080 Laptop GPU；GPU 使用 AMP；CPU 单线程；输入为一个 4 秒试次；每项预热后重复测量。模型延迟不含磁盘 I/O。下表端到端近似值为 CPU 信号处理时间加模型前向时间，未计 CPU→GPU 拷贝和在线缓冲等待。

### 6.1 单试次延迟

| 数据集 | 模型 | 信号处理 CPU | GPU 网络前向 | 近似 GPU 端到端 | CPU 网络前向 | 近似 CPU 端到端 |
|---|---|---:|---:|---:|---:|---:|
| BCIC2a | EEG-Conformer | 0.96 ms | 3.22 ms | **4.18 ms** | **3.55 ms** | **4.51 ms** |
| BCIC2a | FBCNet | 4.58 ms | **0.30 ms** | 4.88 ms | 5.06 ms | 9.64 ms |
| BCIC2a | SE-MHAF-Conformer V1 | 0.96 ms | 4.62 ms | 5.57 ms | 9.33 ms | 10.29 ms |
| BCIC2a | SE-MHAF-Conformer V2 | 4.58 ms | 3.36 ms | 7.94 ms | 11.15 ms | 15.73 ms |
| BCIC2a | SE-MHAF-Conformer Final | 4.58 ms | 3.87 ms | 8.46 ms | 7.07 ms | 11.65 ms |
| BCIC2b | EEG-Conformer | 0.83 ms | 3.31 ms | **4.14 ms** | **2.42 ms** | **3.25 ms** |
| BCIC2b | FBCNet | 4.42 ms | **0.29 ms** | 4.72 ms | 4.90 ms | 9.33 ms |
| BCIC2b | SE-MHAF-Conformer V1 | 0.83 ms | 4.62 ms | 5.44 ms | 5.97 ms | 6.80 ms |
| BCIC2b | SE-MHAF-Conformer V2 | 4.42 ms | 3.63 ms | 8.05 ms | 8.60 ms | 13.03 ms |
| BCIC2b | SE-MHAF-Conformer Final | 4.42 ms | 7.06 ms | 11.49 ms | 11.13 ms | 15.56 ms |

FBCNet 的纯网络推理最快，但需要九次带通滤波；V1 和 EEG-Conformer 只需一次宽带处理，V2/最终版因复用完整 FBC 分支也需要九频带输入。因此，在当前 SciPy CPU 滤波实现下，单试次端到端延迟由 EEG-Conformer 最低。若频带数据在采集端并行生成、使用流式滤波状态或部署到优化后的 DSP，FBCNet、V2 和最终版的端到端延迟可进一步降低。

### 6.2 GPU 批量吞吐量（batch=64）

| 数据集 | EEG-Conformer | FBCNet | SE-MHAF V1 | V2 | Final |
|---|---:|---:|---:|---:|---:|
| BCIC2a | 10,191 trials/s | **11,403 trials/s** | 5,497 trials/s | 5,304 trials/s | 6,789 trials/s |
| BCIC2b | 18,678 trials/s | **27,221 trials/s** | 12,683 trials/s | 12,022 trials/s | 7,955 trials/s |

这些吞吐量只衡量已准备好模型输入后的网络前向，不代表实时脑控系统能达到相同控制频率。真实系统的主要延迟至少还包括 4 秒信号窗、滑窗步长、采集缓存、滤波、置信度平滑和游戏通信。

## 7. 综合评价

### FBCNet

优势：两套数据集均提供很强且稳定的主干；参数、检查点、显存和纯 GPU 推理成本最低，适合嵌入式部署。

代价：九频带滤波是不可忽略的 CPU 前处理成本；小 batch 和长 patience 会增加训练总时间；B02 等弱受试者仍可能失效。

结论：**当前最合适的资源受限部署候选，也是最终版不可破坏的主干**。部署前应将滤波器组改造成保留状态的流式 SOS 滤波，避免每个滑窗从头滤波。

### EEG-Conformer

优势：BCIC2b 与 FBCNet 几乎持平，并取得略高 κ；单宽带前处理简单，当前单试次端到端延迟最低；BCIC2b 上训练总时间最短。

代价：BCIC2a 四分类无增强条件下表现明显落后；参数和显存高于 FBCNet；纯 GPU batch=1 延迟受 kernel launch 开销影响。

结论：**适合强调简单输入链和二分类性能的方案**。应进一步按官方实现加入分段重构增强并进行多种子实验。

### SE-MHAF-Conformer V1

优势：BCIC2b 平均准确率与两条基线相差不大，并在 B01/B02/B07 上取得最高值；结构具备多尺度、通道重标定和头间融合的研究价值。

代价：参数、检查点、显存和推理成本最高；两套数据集均未取得总体最优；当前没有消融证据证明每个新增模块的净贡献。

结论：**目前应定位为待优化研究模型，而不是默认部署模型**。下一步必须做去 SE、去层间注意力关联、去头融合、降低 embed dimension/depth 的消融。

### SE-MHAF-Conformer V2

优势：以 FBCNet logits 为稳定主路径，MHAF 分支只学习残差；零初始化融合保证训练起点与 FBCNet 一致。V2 在 BCIC2b 达到 80.18%，为本轮最高；相对 V1 在两套数据集均提升约 4.5–4.9 个百分点，同时参数量降至约 6.5%–7.3%。

代价：在 BCIC2a 上仍比 FBCNet 低 10.46 个百分点；需要九频带预处理和双分支前向，端到端延迟及训练时间高于单独 FBCNet；B08/B09 出现负迁移，说明残差分支仍可能对强基线造成干扰。

结论：V2 是最终版连续时间专家的有效预训练来源，但已不再是当前总体最优方案。

### SE-MHAF-Conformer Final

优势：独立训练并冻结 FBC 主干，避免联合训练破坏强基线；零初始化残差使起点严格等于 FBCNet；逐频带 SE、logvar MHAF 和连续时间 MHAF 提供互补信息；5% 验证增益门槛允许证据不足的受试者自动回退。最终版在 BCIC2a/2b 分别达到 70.55%/80.81%，是当前两套协议下平均准确率最高的模型。

代价：需要已训练的 FBC 主干，2b 还需要 V2 时间专家；参数、检查点、显存和延迟高于 FBCNet/V2；受试者路由增加了训练与部署元数据管理复杂度。

结论：**当前准确率优先的推荐模型**。低延迟或极低资源部署仍应选择 FBCNet；最终版后续最重要的验证是多随机种子和嵌套交叉验证，而不是继续根据测试结果调权。

## 8. 有效性限制

1. 每个数据集只运行了一个固定随机种子，标准差反映受试者差异而非重复训练不确定性；
2. 各模型没有进行等预算超参数搜索，默认 epochs、batch 和 patience 不同；
3. BCIC2a 四分类和 BCIC2b 二分类不能直接按准确率判断哪个数据集更容易泛化；
4. BCIC2b 本项目是离线 trial 分类，不是原比赛的连续逐采样因果输出；
5. 单试次延迟是已获得完整 4 秒信号窗后的计算时间，不能替代真实控制延迟；
6. 没有测量采集设备、LSL、网络通信和迷宫游戏主循环的开销。

## 9. 产物与复现

- BCIC2a 指标：`outputs/bcic2a_benchmark.json`
- BCIC2b 指标：`outputs/bcic2b_benchmark.json`
- BCIC2a V2 指标：`outputs/bcic2a_se_mhaf_v2.json`
- BCIC2b V2 指标：`outputs/bcic2b_se_mhaf_v2.json`
- BCIC2a Final 指标：`outputs/bcic2a_se_mhaf_final.json`
- BCIC2b Final 指标：`outputs/bcic2b_se_mhaf_final.json`
- 成本明细：`outputs/model_costs.json`
- BCIC2b 源文件校验值：`data/processed/bcic2b/metadata.json`
- 最佳检查点：`outputs/checkpoints/<dataset>/<model>/<subject>.pt`

```powershell
.\.venv\Scripts\python.exe scripts\profile_models.py --output outputs\model_costs.json
.\.venv\Scripts\python.exe scripts\benchmark_models.py --dataset bcic2a --models se_mhaf_conformer_v2 --output outputs\bcic2a_se_mhaf_v2.json
.\.venv\Scripts\python.exe scripts\benchmark_models.py --dataset bcic2b --models se_mhaf_conformer_v2 --output outputs\bcic2b_se_mhaf_v2.json
.\.venv\Scripts\python.exe scripts\benchmark_final_model.py --dataset bcic2a --baseline outputs\bcic2a_benchmark.json --subjects all --logvar-only --min-validation-gain 0.05 --output outputs\bcic2a_se_mhaf_final.json
.\.venv\Scripts\python.exe scripts\benchmark_final_model.py --dataset bcic2b --baseline outputs\bcic2b_benchmark.json --v2 outputs\bcic2b_se_mhaf_v2.json --subjects all --min-validation-gain 0.05 --output outputs\bcic2b_se_mhaf_final.json
.\.venv\Scripts\python.exe -m pytest -q
```
