# LK-Mini-EEG16 在线运动想象系统：调研依据与实施工作流

更新日期：2026-08-31

## 1. 文档目的

本文记录 LK-Mini-EEG16 接入 BCI Maze 的资料依据、技术决策、当前仓库缺口、阶段任务和验收条件。它是开发与实验的共同基线；详细的戴帽、接线、OpenBCI GUI、Python 命令和被试操作步骤见后续的《LK-Mini-EEG16 运动想象实验操作手册》。

## 2. 已核实的设备事实

厂商在 2026-08-14 通信协议及 Python 示例中明确说明：

- LK-Mini-EEG16 兼容 OpenBCI Cyton+Daisy 通信协议；
- 设备使用 Wi-Fi，厂商示例通过 BrainFlow 的 `CYTON_DAISY_WIFI_BOARD` 接入；
- 示例默认设备 IP 为 `192.168.4.1`、端口为 `12345`；
- 采样率命令为 `~6=250 Hz`、`~5=500 Hz`、`~4=1000 Hz`；
- 通道 1-8 使用 `1`-`8` 编码，通道 9-16 使用 `QWERTYUI` 编码；
- EEG 数据为 24 位有符号大端数；16 通道通过相同采样序号的两帧拼接；
- `b` 开始传输，`s` 停止传输；阻抗测试使用 31.5 Hz 测试信号；
- 厂商 Python 示例使用 BrainFlow 完成连接、通道配置、实时显示和 CSV 导出。

厂商原始资料保存在 `vendor/LK-Mini-EEG16/`，不把其中的示例代码直接并入项目源码：示例适合验证连接，但没有 MI 提示、严格事件标注、受试者级模型训练、在线拒识和实验日志。

## 3. 外部项目、论文和官方资料调研

### 3.1 设备接入与实时数据

| 来源 | 可复用结论 |
| --- | --- |
| [BrainFlow Python examples](https://brainflow.readthedocs.io/en/stable/Examples.html) | 官方推荐的会话生命周期是 `prepare_session -> start_stream -> get_board_data -> stop_stream -> release_session`；必须用 `try/finally` 释放设备。 |
| [BrainFlow supported boards](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html) | BrainFlow 提供真实板卡、Synthetic Board 和 Playback Board，适合把硬件接入与算法测试解耦。 |
| [OpenBCI GUI repository](https://github.com/OpenBCI/OpenBCI_GUI) | OpenBCI GUI 可通过 UDP、OSC、LSL 和 Serial 向其他程序发送数据。 |
| [OpenBCI Networking Widget](https://docs.openbci.com/Software/OpenBCISoftware/GUIWidgets/#networking) | LSL 适合把 GUI 中的时间序列同步发送给 Python；Python 端使用 `pylsl`。 |
| [OpenBCI Networking Test Kit](https://github.com/OpenBCI/OpenBCI_GUI/tree/master/Networking-Test-Kit/LSL) | 官方提供 LSL Python 接收示例，可按 `type=EEG` 解析时间序列。 |

由此确定两种数据源：

1. **BrainFlow 直连（正式采集推荐）**：Python 独占设备连接，减少中间环节，能够配置采样率、增益和通道。
2. **OpenBCI GUI -> LSL（调试推荐）**：GUI 负责设备连接和波形观察，Python 只订阅 LSL。这样小组成员能一边看 GUI，一边验证 Python 收到的数据。

同一时刻不能让 OpenBCI GUI 和 Python BrainFlow 同时独占连接设备。需要同时使用时，应让 GUI 独占设备，再通过 LSL 转发。

### 3.2 电极与硬件配置

| 来源 | 可复用结论 |
| --- | --- |
| [OpenBCI EEG setup](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EEGSetup/) | EEG 通道使用共同参考；SRB2 通常连接耳垂参考电极，BIAS 用于共模噪声抑制。 |
| [OpenBCI ExG setup](https://docs.openbci.com/GettingStarted/Biosensing-Setups/ExGSetup/) | Cyton/Cyton+Daisy 的 EEG 通常采用 SRB2 共同参考；BIAS 有助于抑制 50/60 Hz 共模干扰。 |
| [OpenBCI GUI Hardware Settings](https://docs.openbci.com/Software/OpenBCISoftware/GUIWidgets/#hardware-settings) | EEG 通道应使用 Normal 输入，SRB2 打开；PGA 过高且接触不良时容易出现 railed。 |
| [BCI Competition IV](https://www.bbci.de/competition/iv/) | 2a 使用覆盖感觉运动区的 22 个 EEG 通道，2b 使用 C3、Cz、C4，采样率均为 250 Hz。 |

首轮真实 MI 实验采用 8 个运动区通道：`FC3, FC4, C3, Cz, C4, CP3, CPz, CP4`。这是从感觉运动皮层周围取样的工程折中，覆盖左右半球与中线区域。若连接 16 个 EEG 电极，可扩展为：

`Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, Cz, C4, C6, CP3, CP1, CPz, CP2, CP4`

接线映射必须固定写入每次实验的元数据，不能只记录“CH1-CH8”。

### 3.3 MI 采集范式与分类基线

| 来源 | 可复用结论 |
| --- | --- |
| [MNE CSP motor imagery example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html) | CSP 提取空间方差特征，再连接线性分类器，是可复现的 MI 基线。 |
| [MOABB cross-session MI example](https://moabb.neurotechx.com/docs/auto_examples/paradigm_examples/plot_cross_session_motor_imagery.html) | CSP+LDA 和黎曼几何管线是常用的左右手 MI 基线；跨会话评估应与同会话交叉验证分开报告。 |
| [pyRiemann MI example](https://pyriemann.readthedocs.io/en/latest/auto_examples/biosignal-mi/plot_single.html) | 协方差、切空间和正则化 CSP 是小样本 MI 的成熟选择。 |
| [FBCSP paper](https://doi.org/10.3389/fnins.2012.00039) | 运动想象常用 4-40 Hz 滤波器组；CSP 对受试者特异频段和时间窗敏感。 |
| [EEGNet paper](https://arxiv.org/abs/1611.08024) | EEGNet 是紧凑深度学习基线，但仍需要足够数据和严格验证，不能替代小样本传统基线。 |

### 3.4 GitHub 与社区经验的取舍

| 来源 | 评估与处理 |
| --- | --- |
| [OpenBCI Documentation issue #139](https://github.com/OpenBCI/Documentation/issues/139) | OpenBCI 维护者记录了旧版 Motor Imagery 教程在论坛中长期存在未解决问题，后来决定弃用。结论是不能把该教程当作当前 GUI/设备的操作依据，本项目只复用经官方文档核实的硬件和 Networking 流程。 |
| [OpenBCI 社区 MI 教程讨论](https://openbci.com/forum/index.php?p=/discussion/3202/neuropype-motor-imagery-tutorial-questions) | 社区问题说明版本、数据格式和商业依赖会使旧教程难以复现；可用于发现风险，不能替代厂商协议、采样率核对和本机测试。 |
| [octopicorn/bcikit](https://github.com/octopicorn/bcikit) | 早期 OpenBCI 项目包含模拟源、提示、滚动窗、滤波和可视化等完整链路思想，但基于 Python 2.7 且 README 中多项 MI 功能仍列为待实现，因此不直接引入代码。 |
| [Ramiz03S/BCI](https://github.com/Ramiz03S/BCI) | 展示公开数据上的二分类带通+CSP+SVM流程，可作为算法结构参考；它不包含 LK-Mini 实机接入，也不能证明本项目真实被试有效。 |

社区和个人仓库中常见的“实时”“75%-85%”等自报指标未必采用独立会话、拒识或真实在线协议。本项目不复制这些数值作为验收目标；有效性仍以本设备的采样率/信号质量、分组交叉验证、最后 block 时序留出和跨会话复测为准。

项目实施顺序确定为：

1. 先做左手/右手二分类，使用 CSP+LDA 验证设备能否采到可区分 MI；
2. 再比较 FBCNet、现有 Final 模型和可选黎曼几何模型；
3. 二分类稳定后，才扩展双脚或舌头，进行三/四分类；
4. 公开数据集 checkpoint 只用于代码验证，不直接声称能识别 LK-Mini-EEG16 的真实被试。

## 4. 当前仓库问题

### P0：公开实验复现材料不完整

README 引用了正式结果 JSON、处理后数据和 checkpoint，但这些目录被 Git 忽略。远端克隆无法直接复核 70.55%/80.81%，也无法运行原有 Unity 联调脚本。

后续措施：发布可公开的最终指标、配置、文件校验值和 checkpoint 下载说明；真实受试者原始 EEG 与个人模型不得提交公共仓库。

### P0：原 Unity 联调不等于在线脑控

`run_bci_maze_integration.py` 回放 BCIC2a 的完整 4 秒离线 trial，并在第一次有效移动后结束。它只能验证模型输出到 Unity 的 UDP 通路，应明确标记为离线回放冒烟测试。

### P0：缺少实时设备层

当前没有设备抽象、BrainFlow/LSL 数据源、环形缓冲、丢包检测、在线滤波、信号质量检查和实验 marker。

状态：本轮已实现 BrainFlow/LSL/回放/合成数据源、录制器、窗口处理、质量检查与 MI trial 标签；真实设备的持续采样率、断流和丢包仍须在实验室验收。

### P0：现有四类映射不能保证迷宫通关

当前左手/右手/双脚分别映射 left/right/up，舌头不移动，没有 down。随机迷宫的正确路径可能包含向下移动。首版游戏应使用相对转向：左手=左转、右手=右转、自动前进或确认，低置信度=不动作。

### P1：离线处理不能直接当作在线处理

离线整段零相位滤波会使用 trial 结束之后的未来样本，不能原样搬到在线系统。当前 CSP+LDA 基线只在最近一个已经完整取得的 4 秒历史窗内执行同一套零相位滤波，不会读取判决时刻之后的数据；代价是每次判决至少需要完整窗口。后续若缩短到亚秒级连续控制，应改成有状态因果滤波，并重新训练同一处理链。两种方案都必须保持训练与在线处理一致，并配合固定校准、滑动窗口、置信度平滑、拒识和指令冷却。

### P1：测试覆盖不足

现有测试主要检查预处理和模型张量形状。需要增加设备命令构造、合成数据源、MI trial 切片、训练/保存/回载、在线窗口和协议测试，并为 Unity 增加 EditMode/PlayMode 测试。

状态：本轮已补充 Python 设备命令、合成源、录制切片、质量检查、CSP+LDA 保存回载、在线缓冲和拒识测试；Unity 自动化测试仍待补充。

## 5. 目标架构

```text
LK-Mini / OpenBCI GUI LSL / 文件回放 / 合成信号
                    |
                    v
       统一 EEG 数据源 + 时间戳 + 环形缓冲
                    |
                    v
       通道检查 -> 同构窗滤波 -> 坏段/伪迹判断
                    |
          +---------+---------+
          |                   |
       MI 提示采集          在线滑窗
          |                   |
       trial 数据集       受试者模型+拒识
          |                   |
       CSP-LDA训练          控制策略
          |                   |
          +---------+---------+
                    |
                 Unity
```

## 6. 分阶段工作流与验收

### 阶段 A：设备与通道验证

- 实现 BrainFlow、LSL、回放和合成数据源；
- 固定 CH 到 10-20 位置映射；
- 显示实时波形、RMS、峰峰值、漂移、50 Hz 干扰和坏道状态；
- 使用 OpenBCI GUI 的逐通道阻抗模式检查接触质量。

验收：连续采集 5 分钟无异常退出；采样率误差可解释；眨眼、咬牙仅在合理通道产生明显伪迹；所有实际通道都有变化且没有长时间平线/饱和。

### 阶段 B：MI 提示与数据采集

- 首版类别为 `left_hand`、`right_hand`；
- 每个 trial：准备 2 秒、提示 1 秒、运动想象 4 秒、休息随机 2-3 秒；
- 每类至少 40 个有效 trial，建议分 4 个 block；
- 保存原始连续数据、trial 数组、标签、时间戳、通道名、采样率和实验配置；
- 操作员可实时标记伪迹 trial。

验收：trial 数量平衡；每次提示与数据切片时间一致；暂停和提前退出仍能安全保存已采数据。

### 阶段 C：受试者模型

- 先训练 CSP+LDA；
- 使用分层分组交叉验证，并另外保留时间上最后一个 block 做时序留出验证；
- 输出准确率、Macro-F1、混淆矩阵和每折结果；
- 模型文件包含通道顺序、采样率、滤波参数、时间窗和类别映射。

验收：模型能严格回载；通道顺序或采样率不匹配时拒绝推理；结果明显高于机会水平后才进入脑控联调。

### 阶段 D：实时分类

- 使用与训练一致的 4 秒窗口；
- 0.5-1 秒更新一次；
- 连续多窗概率平滑；
- 置信度不足时输出 `unknown`，不发送移动；
- 保存每次窗口的概率、决定、延迟和信号质量。

验收：离线回放与在线路径输出一致；断流、低质量、通道缺失不会产生控制命令；停止后设备会话被释放。

### 阶段 E：Unity 与人体实验

- 改为相对转向/岔路选择；
- 先使用合成预测和文件回放，再接真实 EEG；
- 同时报告分类指标、在线拒识率、通关率、通关时间和路径效率；
- 区分公开数据集、自采离线、在线分类和游戏表现。

## 7. 数据与隐私规则

- 原始 EEG、受试者姓名、联系方式、同意书和个体 checkpoint 不提交公共 Git；
- Git 中只保存匿名示例、代码、配置模板、汇总指标和校验值；
- 受试者可随时退出，退出不影响已获得的其他权益；
- 采集时避免驾驶、操作危险设备或在明显疲劳/不适状态下继续；
- 本系统是研究原型，不用于医疗诊断。

## 8. 首个可交付里程碑

首个里程碑定义为：

1. LK-Mini-EEG16 的 8 个运动区通道在 Python 中稳定显示；
2. 完成一名被试左右手 MI 各 40 个有效 trial；
3. CSP+LDA 训练、保存、回载和离线评估全部通过；
4. 在线程序每 0.5-1 秒输出平滑概率，低置信度拒识；
5. 文件回放和真实设备使用同一推理代码；
6. Unity 能接收左右转向和 `unknown`/停止状态。

达到该里程碑后，再决定是否扩展到四分类和深度模型。
