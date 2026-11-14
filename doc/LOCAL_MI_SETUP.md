# 本机 MI 程序运行说明

新电脑安装请使用[从 GitHub 克隆到设备采集的完整指南](FRESH_PC_SETUP.md)。下文记录现有工作站的环境路径和历史测试。

项目路径：`D:\xht\BCI-Maze game`。

SE-MHAF 十六通道公开预训练模型已保存在本机。短轮采集默认左右手各 10 次，提示流程约 3 分 45 秒；个人微调与 Unity 路口控制使用 [SE-MHAF 真机流程](SE_MHAF_LIVE.md)。下文 `train` 与 `.joblib` 命令保留为 CSP＋LDA 对照入口；启动器会为 SE-MHAF 自动选择 `.venv` 中的 PyTorch 环境。

## 已配置环境

- Anaconda 2024.10-1 已恢复至原路径 `C:\Users\48811\anaconda3`，基础解释器为 Python 3.12.7 64 位。
- MI 独立解释器：`.venv-mi\Scripts\python.exe`，使用该基础解释器及项目独立的 site-packages。
- 依赖安装依据 `requirements-mi.txt`；BrainFlow 5.22.2、pylsl 1.18.2、NumPy 1.26.4、SciPy 1.13.1、scikit-learn 1.5.1、Matplotlib 3.9.2。
- BrainFlow 和 LSL 本地动态库可加载；恢复后 `.venv-mi` 的 `pip check` 通过。
- 研究环境 `.venv` 已恢复 PyTorch 2.5.1+cu121；RTX 4080 Laptop GPU 计算与依赖检查通过。
- 无需管理员权限、激活脚本或 CUDA。BCIC 深度学习研究脚本继续使用原研究环境。
- 启动器在子进程内设置源码路径，不需要为 CSP+LDA 安装 PyTorch/MNE。

## 接线约定

CH1-CH8 保持原线束映射：`FC3,FC4,C3,Cz,C4,CP3,CPz,CP4`。

CH9-CH16 按小组布局增加：`Fz,FC1,FCz,FC2,C5,C6,CP1,CP2`。

参考与 BIAS 使用独立接口，物理接线按 LK-Mini 厂商定义确认。完整戴帽和 MI 方法见 [实验手册](LK-Mini-EEG16运动想象实验操作手册.md)。

## 日常启动

在项目目录打开 PowerShell。直接双击 `bci-mi.cmd` 会打开 20 秒仿真波形，标题标注 synthetic。连接设备、采集和训练使用下面的命令。

```powershell
# 硬件十六路检查；先关闭其他采集软件
.\bci-mi.cmd check --backend brainflow --duration 60

# 熟悉中文被试提示，数据属于练习用仿真信号
.\bci-mi.cmd collect --subject PRACTICE --backend synthetic --trials-per-class 2 --blocks 1

# 正式采集：仅在佩戴、参考/BIAS、波形质量已核查后运行
.\bci-mi.cmd collect --subject S01 --backend brainflow --fullscreen

# 把 session.npz 替换为采集结束打印的文件名
.\bci-mi.cmd train data/recordings/S01/session.npz --subject S01 --output outputs/subject_models/S01/core8_16ch.joblib

# 使用同一被试、新布局训练的模型
.\bci-mi.cmd online --model outputs/subject_models/S01/core8_16ch.joblib --backend brainflow --plot

# 自动验收，不需要设备
.\bci-mi.cmd validate --output-dir outputs/lk_mini_16_workflow/core8_preserved
```

在线入口通过 `--model` 指定模型文件，并校验被试模型的采样率和通道顺序。Fz 开头的前一版十六路布局与当前布局不同，迁移后需重新采集并训练。合成验收模型用于验证软件流程，真人实验使用对应被试与布局的采集模型。

## 自动重连与传输检查

BrainFlow 入口默认启用自动重连。连续 3 秒没有有效样本，或检测到设备包序号跳变、传输异常时，程序结束旧采集会话并重新连接。原生采集库运行在独立工作进程中；清理完成后才启动下一会话，超时未退出的工作进程由程序终止。设备重新可达后，检查、提示采集和在线分类继续运行。按 Ctrl+C 或关闭操作窗口结束当前任务。

恢复后数据使用新的段编号。检查文件保留各段数据、时间戳和恢复事件；提示采集丢弃跨段的试次，已完成试次继续保留；在线分类重新积累完整信号窗口和平滑历史，再输出有效判断。

日常入口共用 `--no-data-timeout`、`--connect-timeout`、`--max-retries` 等参数。完整参数、分段格式与验收条件见 [连接恢复与传输验证](LK_CONNECTION_RECOVERY.md)。

```powershell
# 默认自动恢复；保存 60 秒检查记录
.\bci-mi.cmd check --backend brainflow --duration 60 --output outputs/lk_check/capture.npz

# 未佩戴状态下，记录 300 秒设备传输与恢复情况
.\.venv-mi\Scripts\python.exe scripts/validate_lk_connection.py --duration 300 --wear-state unworn --output-dir outputs/lk_connection

# 单次会话对照：保留收数记录并检查包序号
.\bci-mi.cmd check --backend brainflow --no-auto-reconnect --duration 60 --output outputs/lk_check/single_session.npz
```

## 重建环境

本机已完成这一步。如日后删除环境，可在项目根目录运行：

```powershell
py -3.11 -m venv .venv-mi
.\.venv-mi\Scripts\python.exe -m pip install -r requirements-mi.txt
.\.venv-mi\Scripts\python.exe -m pip check
```

使用 Python 3.12 时，将 `py -3.11` 换为 `py -3.12`；也可使用已安装的 Python 完整路径创建环境。

独立环境的 MI 回归测试：

```powershell
& .\.venv-mi\Scripts\python.exe -m pytest (Get-ChildItem tests/test_lk_mini*.py).FullName -q
```

## 当前恢复工作状态

环境恢复与自动重连实现已完成，完整回归套件 105 项通过，覆盖模型、采集、重试、超时、工作进程退出、分段保存和消费者处理；合成采集、训练和在线回放流程验收通过。未佩戴实机测试完成 300 秒连续采集，收到 16 × 74388 点，包含首次建连等待的整段吞吐为 247.96 点/秒，断流与段内连续性错误均为 0。另一次 120 秒测试在第 20、70 秒主动停止设备发送，两次均自动恢复，接收间隔分别为 8.482、8.499 秒，数据保存为三个独立段。记录位于 `outputs/connection_20261029/`，方法与结果见[连接恢复与传输验证](LK_CONNECTION_RECOVERY.md)。佩戴后的电极质量检查、带标签运动想象采集与个人模型评估分别按实验手册开展。

## 历史设备诊断（2026-09-05）

当时已通过 BrainFlow 接收设备全部 16 路。控制请求发往设备 `192.168.4.1` 的 HTTP 端口，电脑监听 TCP `12345` 接收设备回传数据。最初 5 秒检查收到 1260 个样本，原始接收时间估算约 252.9 Hz；另一次原始数据诊断中，744 个样本的 743 次包序号增量全部为 1，包含正确的 255 到 0 循环。

原始 Wi-Fi 时间戳呈批量到达。程序通过设备包序号检查样本连续性，并保存等间隔样本时间轴；原始接收时间另存在检查 NPZ 的 `received_timestamps`。平均吞吐量按整个测量时长计算，同时保留批量接收的时间特征。

最初检查时尚未完整佩戴；用户之后已完成佩戴并重启设备。其中一次正式 BrainFlow 检查取得 `16 × 15012` 点，60.16 秒，整段吞吐 249.55 点/秒，包序号连续。结束时有 2 路 bad、4 路 warning，下一次初始化再次超时；当时仍待完成持续连接与佩戴信号质量复测。

本机使用厂商配套 USB 路由器的 STA 方式，内置无线网卡保持校园网连接。用户随后确认采集盒曾熄灭，重新开机后能连接，但 30 秒检查仍因包序号跳变中断。详细网络拓扑、原始通信对照测试与待排查原因见 [断流诊断记录](LK_TRANSPORT_DIAGNOSIS_20260905.md)。该阶段独立环境 MI 测试为 51 项通过，此前完整套件为 69 项通过。对应修复与实验事项见 [代码审查记录](MI_CODE_REVIEW.md)。
