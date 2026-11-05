# SE-MHAF 十六通道采集、个人微调与迷宫控制

当前链路采用 SE-MHAF Final logvar 结构，输入为 LK-Mini-EEG16 的 16 路脑电，输出为左手、右手运动想象概率。公开预训练模型保存在 `outputs/pretrained/lk_semhaf16.pt`，个人微调模型保存在 `outputs/subject_models/<subject>/semhaf16.pt`。CSP＋LDA 保留为同一批个人数据上的对照算法。

## 本机入口

在仓库根目录运行 `bci-mi.cmd`。采集、检查和 CSP＋LDA 使用 `.venv-mi`；`pretrain`、`finetune` 和加载 `.pt` 的 `online` 自动使用包含 PyTorch 的 `.venv`。本机 PyTorch 为 2.5.1+cu121，GPU 为 RTX 4080 Laptop。

```powershell
# 检查实际佩戴后的 16 路信号
.\bci-mi.cmd check --backend brainflow --duration 60

# 短轮采集：左右手各 10 次、2 组，提示流程约 3 分 45 秒
.\bci-mi.cmd collect --subject S01 --session calibration01 --backend brainflow --preset short --fullscreen

# 用公开预训练模型进行个人微调，并保存独立留出结果
.\bci-mi.cmd finetune data/recordings/S01/calibration01.npz --subject S01 --device auto

# Unity 打开 SampleScene 并进入 Play 后，启动路口控制
.\bci-mi.cmd online --model outputs/subject_models/S01/semhaf16.pt --backend brainflow --device auto --unity-host 127.0.0.1 --plot
```

再次采集使用新的 `--session` 名称。若需要追加数据，按采集先后顺序向 `finetune` 传入多个文件；`--holdout-session` 可指定整次独立采集作为最终测试。

```powershell
.\bci-mi.cmd finetune data/recordings/S01/calibration01.npz data/recordings/S01/calibration02.npz --subject S01 --holdout-session calibration02
```

## 采集时长与数据划分

| 预设 | 每类次数 | 组数 | 提示流程预计时间 |
| --- | ---: | ---: | --- |
| `short`（默认） | 10 | 2 | 3 分 45 秒 |
| `standard` | 20 | 4 | 7 分 55 秒 |
| `full` | 40 | 4 | 14 分 15 秒 |

每次试次包含准备 2 秒、提示 1 秒、想象 4 秒、随机休息 2–3 秒；预热 5 秒，组间休息 30 秒。表中时间为随机休息均值下的提示时序。戴帽、调节电极、连接等待和丢弃试次后的追加采集另计。显式 `--trials-per-class` 与 `--blocks` 覆盖预设。

完整保留一轮 short 的 20 个试次时，个人微调使用 6 个训练试次、4 个早停验证试次和末组 10 个最终测试试次。报告记录每个划分的索引、类别数和 `small_sample` 状态。standard 的对应划分为 20/10/10，full 为 40/20/20。最终测试试次保持独立；保存模型保留验证选出的参数。出现伪迹或中断时，程序报告有效次数；某个划分类别不足时需要追加采集。

采集文件记录微伏单位、通道顺序、采样率、增益、参考设置、来源、标签来源、试次、人工伪迹标记和连接分段。实际参考、BIAS 位置可通过 `--reference-location`、`--bias-location` 保存；未填写时记为 `unspecified`。

## 公开预训练与预处理

BCIC2a 原始 22 路数据按标准帽位映射，抽取并重排为 `FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2`。保留左右手标签及无伪迹试次，采用原记录 trial 开始后 2–6 秒的 1000 点窗口。映射依据为 [BCIC2a 官方导联图](https://www.bbci.de/competition/iv/desc_2a.pdf)及 MOABB 的 BCIC2a 通道顺序。实际设备的线束、插口与帽位应与该顺序对应。

公开训练与真机推理共用同一处理函数：微伏输入 → 16 路共同平均参考（CAR）→ 4–40 Hz 的九个 4 Hz 频带 → 每个窗口从零初态进行因果滤波 → 按训练试次计算的逐频带、逐通道统计归一化 → SE-MHAF。在线窗口为 4 秒，每 0.5 秒更新。模型包保存完整预处理、通道、类别和来源信息，加载时逐项校验。

默认使用 A01–A07 的 T 会话进行训练，A08–A09 的 T 会话用于选择训练轮次。E 会话在模型保存后单独评估。16 路、二分类、CAR 模型从固定种子初始化，全部层参与训练；已有 22 路、四分类的按被试检查点保留在原路径。

```powershell
.\bci-mi.cmd pretrain --device cuda --epochs 100 --patience 20 --output outputs/pretrained/lk_semhaf16.pt
.\bci-mi.cmd pretrain --evaluate-only outputs/pretrained/lk_semhaf16.pt --device cuda --evaluation-output outputs/semhaf_20261029/public_evaluation.json
```

个人微调从公开检查点开始，以较小学习率更新参数；均值、标准差只由训练部分计算，验证部分选择早停轮次，最终留出部分报告微调前后的识别结果。多次采集使用同一公开检查点和累积记录重新微调，以保持当前留出划分可追溯。

## 路口决策与断线恢复

游戏通过 UDP `127.0.0.1:7777` 接收指令。默认 `--control-mode junction` 会读取 Unity 状态，在路口取得暂停确认后清空旧窗口，积累当前路口的新数据，再进行质量检查及三次概率平滑。最高概率默认至少 0.65，两类差值至少 0.15。每次路口接受一个稳定决策。

左右手对应路口的左右两个可选方向：T 字口为左转/右转；直行加右侧出口时为直行/右转；左侧出口加直行时为左转/直行。三个出口同时可用时左右手对应左转/右转，直行保留原有按钮与键盘入口。在线图和日志显示本次映射。

动作携带预期游戏状态，由 Unity 主线程在执行前校验；地图重建、位置或暂停状态变化会使旧请求失效。UDP 往返由独立线程处理，采集读取持续运行。退出时先撤销控制意图、清理程序拥有的暂停，再释放采集设备。

连续 3 秒无数据或发生传输异常时，已有采集恢复器自动重建连接。在线逻辑取消旧预约、暂停决策并清空数据窗口、平滑历史；恢复后重新积累完整数据。采集阶段保留已完成试次，丢弃跨断点试次。恢复参数及此前真机停流结果见 [连接恢复记录](LK_CONNECTION_RECOVERY.md)。

使用真实设备或 LSL 运动想象记录训练的个人 SE-MHAF、CSP＋LDA 模型可直接进入游戏控制，包括独立留出结果暂未达到统计阈值的模型。在线日志如实保存 `model_origin`、`calibration_passed` 和 `control_validated`。公共预训练、合成来源及缺少来源信息的旧模型，需要显式加入 `--allow-unvalidated-control`。纯预测可加载受支持的模型。

## 已保存结果

- 公开预训练在第 48 轮早停，选择第 28 轮。验证被试等权平均平衡准确率为 82.32%。
- 独立 E 会话：A01–A07 的被试等权平均准确率为 71.43%；A08–A09 为 74.09%。后两名被试的 T 会话曾用于选择轮次，报告按该条件单独列出。
- 公开预训练模型与默认 short 的合成记录已实际完成 GPU 微调、保存、重载和独立记录在线回放。该模型保存在 `outputs/semhaf_20261029/software_finetuned.pt`，来源为 `synthetic`，个人校准通过状态保持为 false。
- 全部 170 项 Python 测试通过，结果保存在 `outputs/semhaf_live_20261029/pytest.xml`。Unity 实际执行 7 项状态条件检查，并通过真实 UDP 拒绝过期地图代数的请求。
- 既有未佩戴真机记录经过新在线入口回放，588 个窗口均因信号质量被拒绝，接受分类数为 0。
- 60 秒实时合成脑电回放经上述 SE-MHAF 模型输出 3 次路口决策，在种子 42 的 Unity 迷宫中到达终点，共 56 步、17 次转向。期间注入一次采集分段变化，后续决策使用重新积累的数据窗口；确认回执中没有重复路口提交。试验记录为 `outputs/semhaf_20261029/unity_replay/report.json`，输入为合成记录，游戏为实际 Unity Play 实例。
- 采集盒重新开机后，`/board` 查询成功，SDK 选择 `192.168.4.2` 作为监听和回传地址。120 秒未佩戴真机测试收到 16 × 25212 点；第 25、75 秒调用设备 `/stream/stop` 后，两次均由默认无数据检测自动恢复，形成 3 段，段内连续性错误为 0。实际接收时间记录的两次空档为 8.395、8.452 秒，检测到断流至首个恢复数据块分别为 5.391、5.437 秒。
- 上述新真机记录按 SE-MHAF 在线流程共检查 179 个完整窗口，均因信号质量返回 `bad_signal`。另外，每段首个完整窗口执行一次独立 GPU 前向计算，三次输出均为有限值；热身后的两次计算耗时为 8.17、9.72 毫秒。该计算检查保存在 `outputs/semhaf_live_20261029/reconnection/semhaf_capture_analysis.json`，游戏指令数为 0。
- 随后的 180 秒连续采集收到 16 × 44424 点；首次收数等待 2.328 秒，最长收数间隔 0.094 秒，意外断流和段内连续性错误均为 0。记录保存在 `outputs/semhaf_live_20261029/continuous`。
- 正式 `online` 入口加载公开 SE-MHAF 模型并使用 CUDA，完成 120 秒真机测试，收到 16 × 24648 点。首批数据到达后第 25、75 秒调用设备 `/stream/stop`，两次分别在 9.547、9.266 秒后恢复收数，在 13.516、13.266 秒后形成包含 1000 个新样本的完整 4 秒窗口。175 个窗口均返回 `bad_signal`，在线模型前向调用和接受决策数均为 0，跨段数据混入检查通过。记录保存在 `outputs/semhaf_live_20261029/online/live_report.json`。
- 10 月 29 日该轮验收结束时，真人带标签采集尚未开展。10 月 31 日已保存 76 个运动想象完整试次，采集归属和恒定通道检查见[实际采集整理](COLLECTION_20261031.md)；个人模型微调和独立评价待完成。

此前采集盒关闭时，SE-MHAF 真机入口曾完成 45 秒重试并正常退出，设备 `/board` 请求超时，样本数为 0；当时 SDK 日志中的本机地址为 Meta 网卡的 `198.18.0.1`。重新开机后的首次连接和两次自动恢复均使用设备网卡 `192.168.4.2`，沿用原有网络设置。BrainFlow 将所选本机地址用于监听和 `/tcp` 回传配置，实现依据见 [BrainFlow Wi-Fi 初始化源码](https://github.com/brainflow-dev/brainflow/blob/5.22.2/src/board_controller/openbci/openbci_wifi_shield_board.cpp#L56)。

在线测试于本地时间 13:33:40 正常结束；13:34:35 复查时 TCP 12345 已释放，但采集盒再次出现 HTTP 超时和 ICMP 丢包，接收器 `192.168.4.254` 仍可达。设备网卡保持 Up、地址 `192.168.4.2/24`，直连路由存在，采集盒邻居地址解析未完成。随后正式在线入口继续运行 60 秒、自动尝试连接 8 次，样本数为 0；退出后确认 TCP 12345 已释放。厂商手册说明手动长按开关机及供电参数，当前失联原因仍待结合采集盒供电和无线指示状态核实。网络和重试记录为 `outputs/semhaf_live_20261029/postcheck_network.json`、`postcheck_retry/transport.jsonl`，本轮完整验收总表为 `outputs/semhaf_live_20261029/verification_summary.json`。

真机在线恢复验收入口如下；每次使用新的结果目录，脚本会调用正式在线入口并保存传输、分类窗口、故障注入和汇总记录：

```powershell
.\.venv\Scripts\python.exe -u scripts/validate_semhaf_live.py --model outputs/pretrained/lk_semhaf16.pt --output-dir outputs/semhaf_live_check --duration 120 --stop-stream-at 25,75 --wear-state unworn --device cuda
```

日常在线入口可增加 `--transport-log <文件.jsonl>` 记录逐批收数、数据段、窗口重建和退出汇总。恢复验收同时检查注入成功、断流与恢复事件、新段样本、窗口时间范围、有限数值、末尾活跃及接受决策记录。

验证总表在 `outputs/semhaf_20261029/verification_summary.json`。详细指标、训练历史和数据划分分别在 `outputs/pretrained/lk_semhaf16.json`、`outputs/semhaf_20261029/public_evaluation.json` 和 `outputs/semhaf_20261029/software_finetuned.json`。

Unity 联机验证入口为 `uv run scripts/validate_semhaf_unity.py --model <模型.pt> --replay <记录.npz> --output <新结果目录>`。项目提供显式编辑器启动方法 `BciVerification.Start`，可通过 Unity 命令行 `-executeMethod BciVerification.Start` 启动已配置的本地 MCP 服务。普通打开工程保持原有启动行为。
