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

连接 LK-Mini-EEG16 并使用网页采集，请按[新电脑安装与设备采集指南](doc/FRESH_PC_SETUP.md)配置 `.venv-mi`。下面的 `.venv` 安装与公开数据集步骤用于模型研究。

### 1. 安装 Python 依赖

项目要求 Python 3.10 或更高版本。PyTorch 请先按本机 CUDA / CPU 环境从官方渠道安装，再安装其余依赖；requirements.txt 有意没有固定 torch，以免覆盖正确的 CUDA 构建。

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 先按 https://pytorch.org/ 安装适合本机的 PyTorch
pip install -r requirements.txt
pip install -e . --no-deps
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

- 默认生成 15 × 15 的 3D 第一人称迷宫，使用迭代 DFS；
- 起点为 (1, 1)，终点为 (width - 2, height - 2)；
- 自动前进，唯一出口的拐角自动转向；分岔处留出 4 秒选择，有直行出口则到时直行，T 字路口等待左右选择；
- A/D 或左右方向键预约相对左转/右转，S/↓ 掉头；死胡同默认自动掉头，可在设置中关闭；
- Space 暂停，R 重建，H 路线提示，M/V 小地图；屏幕按钮支持同样的操作；
- 外部控制监听 127.0.0.1:7777。

接口和线程语义详见 [Unity BCI Maze 接口说明](Maze-game/Assets/README_BCI_Maze.md)。在线联调：

~~~powershell
python scripts/run_bci_maze_integration.py --host 127.0.0.1 --port 7777
~~~

该脚本读取 BCIC2a A01 的 E 测试数据和默认 Final checkpoint，发送指令并等待 Unity ACK。新版 left/right 为相对转向，up/down 兼容为继续前进/掉头；accepted 表示指令已接受（可能仍在预约），不等于角色立即移动或模型正确。四分类可加 `--label-map configs/label_maps/bcic2a_relative.json`，将双脚映射为掉头。左右手二分类可直接控制新版。

无 EEG 设备时，在 Unity Play 模式运行 `uv run scripts/validate_unity_maze.py`，可复核导航回归和真实 UDP 通关；详见 [3D 迷宫流程与验收](doc/UNITY_FIRST_PERSON_DEMO.md)。

## LK-Mini-EEG16 真实设备与受试者 MI

**新电脑从这里开始：[安装环境、取得配套驱动、连接设备并完成采集](doc/FRESH_PC_SETUP.md)。** 采集使用 `.venv-mi`；厂商 USB 路由器驱动、使用说明书和 OpenBCI GUI 安装包需从设备配套资料包取得。指南分别列出 GitHub 已包含的文件与厂商资料清单，并提供 STA/AP 连接和网页保存步骤。

本地可视化控制台：双击根目录的 `bci-dashboard.cmd`，或运行 `bci-mi.cmd ui`。在浏览器中集中配置数据源、查看十六路波形与信号质量、启动采集和模型任务，并查询及控制 Unity 迷宫。操作流程、质量指标和电极接触检测的实现状态见 [实验控制台说明](doc/CONTROL_CONSOLE.md)。

控制台的“05 / 子实验”提供真实左右握拳对照、带独立静息基线的运动想象采集与信号分析，以及按会话或采集分组评价的 CSP＋LDA 和标签置乱对照。支持已有记录分析、仿真演示与报告图像查看，详见[子实验操作与结果说明](doc/EXPERIMENTS.md)。

2026-10-31 已完成 S01–S04 实际采集，保存 69 个真实握拳和 76 个运动想象完整试次，另有 2 个任务类型待确认的旧校准试次。16:02 所在的握拳记录及随后短想象记录已按操作者确认更正为 S03。逐次清单、恒定通道标记与恢复统计见[实际采集整理](doc/COLLECTION_20261031.md)。

LK-Mini-EEG16 采集链路默认启用全部 16 个 EEG 通道，支持 BrainFlow Wi-Fi 直连、OpenBCI GUI LSL、NPZ 回放和合成信号。SE-MHAF Final 已接入公开预训练、个人微调和游戏路口控制；CSP+LDA 保留为对照。检查窗口显示 16 路波形及逐路质量，采集窗口显示中文动作提示、倒计时和异常电极。完整流程及已有模型见 [SE-MHAF 十六通道真机流程](doc/SE_MHAF_LIVE.md)。

个人微调使用 `outputs/pretrained/lk_semhaf16.pt`。若本地尚无该文件，先按[公开预训练说明](doc/SE_MHAF_LIVE.md#公开预训练与预处理)准备 BCIC2a 数据，再运行：

~~~powershell
.\bci-mi.cmd pretrain --device cuda --epochs 100 --patience 20 --output outputs/pretrained/lk_semhaf16.pt
~~~

~~~powershell
# 1. 连接与通道质量检查
.\bci-mi.cmd check --backend brainflow --duration 60

# 2. 默认短轮：每类 10 次、2 组，约 3 分 45 秒
.\bci-mi.cmd collect --subject S01 --session calibration01 --backend brainflow --preset short --fullscreen

# 3. 从已保存的公开预训练模型进行个人微调
.\bci-mi.cmd finetune data/recordings/S01/calibration01.npz --subject S01

# 4. Unity 进入 Play 后，启动实时识别与路口控制
.\bci-mi.cmd online --model outputs/subject_models/S01/semhaf16.pt --backend brainflow --unity-host 127.0.0.1 --plot
~~~

默认通道顺序为 `FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2`，依次对应 CH1-CH16，保留原 core-8 物理通道，默认 250 Hz。参考 SRB2 与 BIAS 不占 EEG 通道。模型会校验采样率和通道顺序；低置信度、类别概率差距不足或信号坏道时输出 `unknown`，但二分类模型没有可靠的静息检测能力。

从八通道升级保持 CH1-CH8 不动，按表新增 CH9-CH16，再重新采集和训练；旧模型不能直接用于十六路输入。旧八通道记录仍可回放；复现旧布局需显式使用 `--channels 1,2,3,4,5,6,7,8 --channel-names FC3,FC4,C3,Cz,C4,CP3,CPz,CP4`。Mini-EEG08 串口诊断脚本 `test_mini_eeg_connection.py` 不适用于本 Wi-Fi 设备。

本机已配置独立 `.venv-mi` 环境，使用 `bci-mi.cmd` 启动；`bci-mi.cmd demo` 打开仿真波形，`bci-mi.cmd check --backend brainflow` 检查真机。详见 [本机运行说明](doc/LOCAL_MI_SETUP.md)。

BrainFlow 默认自动重连：连续 3 秒无有效样本或出现包序号异常时，程序清理独立采集工作进程并重新建立会话，设备恢复后继续收数。记录文件按连接会话分段；采集程序丢弃跨段试次，在线分类重新积累完整窗口与平滑历史。使用 `--no-auto-reconnect` 运行单次会话对照；参数和传输验收入口见 [连接恢复与传输验证](doc/LK_CONNECTION_RECOVERY.md)。

面向初学者的接线、配置、MI 方法、完整命令、数据格式和排障见 [LK-Mini-EEG16 运动想象实验操作手册](doc/LK-Mini-EEG16运动想象实验操作手册.md)。历史调研见 [在线 MI 系统调研与实施工作流](doc/在线MI系统调研与实施工作流.md)，设备操作以当前手册为准。

无设备时可执行 `python scripts/validate_lk_mini_workflow.py`，验证十六通道检查、合成采集、训练和独立在线回放。报告位于 `outputs/lk_mini_workflow/workflow_report.json`。实际设备采集结果见[实际采集整理](doc/COLLECTION_20261031.md)。

## 测试

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
~~~

测试覆盖预处理、模型构造、形状约束、LK-Mini 设备命令、合成数据源、录制切片、信号质量、CSP+LDA 保存回载以及在线缓冲/拒识。涉及真实数据、真实硬件或 GPU 的实验不属于 pytest 的默认范围。

可在没有 EEG 设备时运行完整的四阶段软件验收。它会依次启动通道检查、合成 MI 采集、受试者模型训练和独立记录的在线回放分类，并验证坏信号不会进入模型：

~~~powershell
python scripts/validate_lk_mini_workflow.py
~~~

结果和中间产物写入 `outputs/lk_mini_workflow/`。该命令使用合成输入检查采集、保存、训练和在线拒识。电极位置、阻抗、运动想象可分性和持续传输分别按实验手册采集现场记录。

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

- 当前实验采用受试者内评价（subject-dependent），跨受试者泛化留待独立实验。
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
