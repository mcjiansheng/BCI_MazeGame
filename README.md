# BCI Maze：运动想象解码与 Unity 迷宫控制

本项目把两部分串成一条可复现实验链路：

1. 使用 Python / PyTorch 对 BCI Competition IV 2a、2b 运动想象 EEG 进行预处理、训练和评估。
2. 使用 Unity 生成可交互迷宫，并把模型预测映射为左、右、上、下移动指令。

当前仓库的主任务是运动想象分类，不是测谎。Box 文件夹中的资料是另一个“基于脑机交互的测谎系统”项目的设计、采集和原型代码，只作为外部参考。详细分析报告仅保留在本地 doc/Box资料阅读与项目参考分析.md，未同步到远端；Box 原文件保存在本地的 tmp/ 目录中，该目录已被 Git 忽略，不应提交或上传。

## 当前实验结果

以下数字来自仓库内 outputs/ 下的已保存实验结果，统计方式为 subject-dependent：每位受试者独立划分训练/验证集，并在独立 E 测试集上评估。默认随机种子为 42，并默认排除伪迹样本。结果更新时间：2026-08-28。

| 数据集 | 模型 | Accuracy | Macro-F1 | Cohen's κ |
| --- | --- | ---: | ---: | ---: |
| BCIC2a | EEG-Conformer | 56.03% | 52.10% | 0.414 |
| BCIC2a | FBCNet | 69.30% | 68.28% | 0.591 |
| BCIC2a | SE-MHAF V1 | 53.93% | 52.44% | 0.386 |
| BCIC2a | SE-MHAF V2 | 58.84% | 57.30% | 0.452 |
| BCIC2a | SE-MHAF Final | 70.55% | 69.70% | 0.607 |
| BCIC2b | EEG-Conformer | 76.57% | 75.43% | 0.537 |
| BCIC2b | FBCNet | 76.81% | 76.25% | 0.536 |
| BCIC2b | SE-MHAF V1 | 75.69% | 74.97% | 0.519 |
| BCIC2b | SE-MHAF V2 | 80.18% | 80.04% | 0.604 |
| BCIC2b | SE-MHAF Final | 80.81% | 80.51% | 0.614 |

这些是本仓库当前配置下的可复现实验结果，不代表论文原始结果，也不应解读为跨受试者泛化性能。完整指标、混淆矩阵和训练配置见：

- [BCIC2a 与 2b 三模型综合实验报告](doc/BCIC2a与2b三模型综合实验报告.md)
- [SE-MHAF-Conformer 最终优化与验证报告](doc/SE-MHAF-Conformer_最终优化与验证报告.md)
- outputs/bcic2a_benchmark.json、outputs/bcic2b_benchmark.json
- outputs/bcic2a_se_mhaf_v2.json、outputs/bcic2b_se_mhaf_v2.json
- outputs/bcic2a_se_mhaf_final.json、outputs/bcic2b_se_mhaf_final.json

## 快速开始

### 1. 安装 Python 依赖

项目要求 Python 3.10 或更高版本。PyTorch 请先按本机 CUDA / CPU 环境从官方渠道安装，再安装其余依赖；requirements.txt 有意没有固定 torch，以免覆盖正确的 CUDA 构建。

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 先按 https://pytorch.org/ 安装适合本机的 PyTorch
pip install -r requirements.txt
~~~

当前验证环境为 Python 3.11、PyTorch 2.5.1+cu121、MNE 1.8.0 和 NVIDIA RTX 4080 Laptop GPU。其他环境仍应重新运行测试和实验。

### 2. 准备 BCIC2a 数据

将官方 GDF 文件放在 data/BCIC2a/：

~~~text
data/BCIC2a/
├── A01T.gdf
├── A01E.gdf
└── ... A09T.gdf / A09E.gdf
~~~

预处理命令：

~~~powershell
python scripts/preprocess_bcic2a.py --raw-dir data/BCIC2a --output-dir data/processed/bcic2a
~~~

默认处理设置是 250 Hz、事件后 2–6 秒、4–40 Hz Chebyshev II 带通滤波，并保留 22 个 EEG 通道、排除 EOG。每个 trial 的主数组形状为 trials × 22 × 1000，同时保存 raw_x、标签 y、伪迹标记 artifact 和元数据。滤波器组包含 4–8、8–12、12–16、16–20、20–24、24–28、28–32、32–36、36–40 Hz。

BCIC2a 标签映射为：

| 标签 | 运动想象 | Unity 指令 |
| ---: | --- | --- |
| 0 | 左手 | left |
| 1 | 右手 | right |
| 2 | 双脚 | up |
| 3 | 舌头 | no movement（不发送移动） |

### 3. 准备 BCIC2b 数据（可选）

本项目使用 BNCI 004-2014 / BCI Competition IV 2b 的 MAT 文件，放在 data/BCIC2b/。预处理默认读取 3 个 EEG 通道 C3、Cz、C4，使用 250 Hz、事件后 3–7 秒窗口，并区分 T 训练会话和 E 测试会话。

~~~powershell
python scripts/preprocess_bcic2b.py --raw-dir data/BCIC2b --output-dir data/processed/bcic2b
~~~

官方数据入口：[BNCI Horizon 2020 004-2014](http://bnci-horizon-2020.eu/database/data-sets/004-2014)。

## 训练与评估

### 基准模型

首次运行完整基准：

~~~powershell
python scripts/benchmark_models.py --dataset bcic2a --processed-dir data/processed/bcic2a --output outputs/bcic2a_benchmark.json
python scripts/benchmark_models.py --dataset bcic2b --processed-dir data/processed/bcic2b --output outputs/bcic2b_benchmark.json
~~~

只运行指定模型或受试者：

~~~powershell
python scripts/benchmark_models.py --dataset bcic2a --models eeg_conformer,fbcnet,se_mhaf_conformer,v2,v3,v3_raw,v3_compact --subjects 1,2
~~~

模型注册表位于 src/bci_maze/models/__init__.py，当前包括 EEG-Conformer、FBCNet、SE-MHAF V1、V2、V3、V3 raw、V3 compact、Final，以及 Final logvar 变体。V3 使用宽带输入和可选的分段重构增强；FBCNet、V2、Final 使用滤波器组输入。

常用默认训练配置：

| 模型 | epochs | patience | batch size | learning rate | 增强 |
| --- | ---: | ---: | ---: | ---: | --- |
| EEG-Conformer / V1 | 200 | 40 | 64 | 2e-4 | 否 |
| FBCNet / V2 | 500 | 100 | 16 | 1e-3 | 否 |
| V3 / V3 raw | 1000 | 200 | 72 | 2e-4 | 是 |
| V3 compact | 500 | 100 | 64 | 2e-4 | 是 |
| Final | 300 | 80 | 64 | 1e-3 | 默认是 |

训练只使用训练集拟合归一化参数，验证集用于选择最佳 epoch，E 会话作为最终测试集。可通过 --include-artifacts 纳入伪迹样本、通过 --no-augment 关闭增强，完整参数以脚本 --help 为准。

### Final 模型

Final 是基于基线 FBCNet 的门控融合模型。仓库已保存的 BCIC2a 结果采用 log-variance 分支，BCIC2b 结果采用 V2 时间专家；命令中需要显式传入对应选项：

~~~powershell
python scripts/benchmark_final_model.py --dataset bcic2a --baseline outputs/bcic2a_benchmark.json --logvar-only --output outputs/bcic2a_se_mhaf_final.json
python scripts/benchmark_final_model.py --dataset bcic2b --baseline outputs/bcic2b_benchmark.json --v2 outputs/bcic2b_se_mhaf_v2.json --output outputs/bcic2b_se_mhaf_final.json
~~~

如需先训练 V2 时间专家，可按脚本参数传入相应 checkpoint。Final 脚本会检查相对 FBCNet 的最低收益阈值；若要严格复核仓库已保存的最终结果，使用：

~~~powershell
python scripts/verify_final_results.py --result outputs/bcic2a_se_mhaf_final.json
python scripts/verify_final_results.py --result outputs/bcic2b_se_mhaf_final.json
~~~

默认 checkpoint 位置为 outputs/checkpoints/{dataset}/，该目录及 JSON/CSV 输出已被 .gitignore 忽略。

## Unity 迷宫与在线控制

Unity 项目位于 Maze-game/，使用 Unity 2022.3.57f1c1。打开场景 [SampleScene.unity](Maze-game/Assets/Scenes/SampleScene.unity) 后点击 Play：

- 默认生成 61 × 61 的完美迷宫，使用迭代 DFS；
- 起点为 (1, 1)，终点为 (width - 2, height - 2)；
- WASD 或方向键移动，R 重新生成迷宫；
- H 显示 BFS 最短路长度，V 切换全图 / 跟随视角；
- 外部控制监听 127.0.0.1:7777。

接口和线程语义详见 [Unity BCI Maze 接口说明](Maze-game/Assets/README_BCI_Maze.md)。在线联调：

~~~powershell
python scripts/run_bci_maze_integration.py --host 127.0.0.1 --port 7777
~~~

该脚本读取 BCIC2a A01 的 E 测试数据和默认 Final checkpoint，把预测类别映射为方向并等待 Unity ACK。它是端到端冒烟测试，不是迷宫通关率评估；墙体阻挡、无效方向或模型预测错误都会导致某次命令未被接受。

## 测试

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
~~~

测试覆盖预处理、模型构造、形状约束和若干训练工具函数。涉及真实数据或 GPU 的实验不属于 pytest 的默认范围。

## 目录结构

~~~text
src/bci_maze/                 预处理、训练和模型实现
scripts/                      预处理、基准实验、验证和 Unity 联调脚本
tests/                        自动化测试
data/BCIC2a/                  本地 GDF 数据（不提交）
data/BCIC2b/                  本地 MAT 数据（不提交）
data/processed/               预处理缓存（不提交）
outputs/                      指标、checkpoint 和联调输出（不提交）
Maze-game/                    Unity 工程
doc/                          实验、算法、模型和资料分析文档
tmp/box-nju-1644522/          Box 下载资料（不提交）
~~~

建议阅读顺序：

1. [预处理与三模型实现说明](doc/BCIC2a预处理与三模型实现说明.md)
2. [BCIC2b 预处理与适配说明](doc/BCIC2b预处理与适配说明.md)
3. [EEG-Conformer、FBCNet、SE-MHAF 模型解析](doc/EEG_Conformer_FBCNet_SE-MHAF_模型解析.md)
4. [最终优化与验证报告](doc/SE-MHAF-Conformer_最终优化与验证报告.md)
5. [Unity BCI Maze 接口说明](Maze-game/Assets/README_BCI_Maze.md)
6. 本地 Box 资料阅读与项目参考分析（未同步到远端）

## 已知边界与安全注意事项

- 当前实验是 subject-dependent，尚未证明跨受试者泛化；不能把单受试者测试结果当作通用控制准确率。
- Unity 默认只监听回环地址且没有鉴权；若改为局域网监听，必须增加访问控制，并同步修改脚本、场景和文档。
- Box 资料包含受试者安排、访谈文本、面部行为记录和其他敏感信息。下载目录在 tmp/ 下且被忽略；不要把原始资料、个人信息或外部项目结果复制进 Git。
- 外部资料中的“测谎”标签协议、模型分数和报告指标没有被本项目验证；尤其不能把资料报告中的 91.6% 直接写成 BCI Maze 的结果。
- 资料中的机器打分脚本含有硬编码的第三方 API 凭据，且会把访谈内容发送到外部服务。本仓库没有执行它们；如果凭据仍有效，应立即撤销/轮换，并改用环境变量或安全密钥管理。

## 参考资料

- [BCI Competition IV 2a](https://www.bbci.de/competition/iv/)
- [BNCI Horizon 2020 004-2014](http://bnci-horizon-2020.eu/database/data-sets/004-2014)
- [EEG-Conformer](https://arxiv.org/abs/2106.11170)
- [FBCNet](https://arxiv.org/abs/2104.01233)
- [MNE-Python 文档](https://mne.tools/stable/)
- [PyTorch 文档](https://pytorch.org/docs/stable/)
