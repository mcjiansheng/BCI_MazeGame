# 新电脑安装与 LK-Mini-EEG16 采集

本流程用于 Windows 10/11 64 位电脑，从 GitHub 克隆项目，安装采集环境，连接 LK-Mini-EEG16，并通过浏览器控制台保存脑电记录。采集、波形监测和 CSP＋LDA 使用 `.venv-mi` 环境；安装内容为 Git、Python 和 `requirements-mi.txt` 中的依赖。

## 1. 准备设备配套资料

克隆仓库会取得采集代码、网页控制台、Python 依赖清单，以及 `vendor/LK-Mini-EEG16/` 中的通信协议和厂商 Python 示例。以下配套文件需要从购买设备时提供的资料包取得；缺少资料包时，向设备销售方或厂商客服索取下表中的具体文件。

| 配套内容 | 文件或识别名称 | 用途与获取方式 |
| --- | --- | --- |
| 厂商使用说明书 | `LK-Mini-EEG16 使用说明书 20260716.pdf`，或对应设备批次的新版 | 向厂商索取，核对接口、参考电极、设备工作模式和配套接收器。该版本第 7–9 页说明 STA/AP 连接和上位机设置。 |
| USB 路由器及 Windows 驱动 | `上位机软件/USB路由器驱动程序/windows/SR9900_SFX.EXE`；资料包另有 `SR9900_drivers_WIN10.rar` | 采用配套 USB 接收器的 STA 方式时使用。安装后网卡名称为 `Corechip SR9900 USB2.0 to Fast Ethernet Adapter`。驱动从厂商资料包取得。 |
| OpenBCI GUI | `openbcigui_v6.0.0-beta.1_windows64.zip` | 厂商手册对应的上位机版本。解压后运行 `OpenBCI_GUI.exe`。厂商另给出了 [OpenBCI 官方下载页](https://openbci.com/downloads)；复现本项目设备配置时优先使用配套版本。 |
| 电极帽、线束、参考与 BIAS 电极、接触材料 | 对应设备的完整采集配件 | 按厂商接口标识接线，头皮通道顺序见[实验操作手册](LK-Mini-EEG16运动想象实验操作手册.md#4-电极位置与接线)。 |

GitHub 仓库不含上述使用说明书、USB 驱动和 GUI 安装包。`vendor/` 中的通信协议 PDF 与使用说明书分别介绍通信格式和设备操作，Python 示例压缩包也不包含 USB 驱动。当前仓库没有厂商驱动的公开下载地址，STA 连接需先取得配套资料包。

BrainFlow 可以直接连接设备。日常使用本项目网页采集时，数据直接进入 Python；OpenBCI GUI 用于独立设备检查，选择 LSL 数据源时则由 GUI 持续转发数据。

## 2. 安装 Git、Python 并克隆 main

安装 [Git for Windows](https://git-scm.com/downloads/win) 和 [Python 64 位 Windows 安装程序](https://www.python.org/downloads/windows/)。采集依赖采用固定版本，推荐 Python 3.11 或 3.12，安装 Python 时保留 pip、Python Launcher 和 Tcl/Tk 支持。

在准备存放项目的目录打开 PowerShell，运行：

```powershell
git clone --branch main --single-branch https://github.com/mcjiansheng/BCI_MazeGame.git
cd BCI_MazeGame
py -3.11 -m venv .venv-mi
.\.venv-mi\Scripts\python.exe -m pip install --upgrade pip
.\.venv-mi\Scripts\python.exe -m pip install -r requirements-mi.txt
.\.venv-mi\Scripts\python.exe -m pip check
```

安装的是 Python 3.12 时，把创建环境命令中的 `py -3.11` 换为 `py -3.12`。没有 `py` 命令时，使用已安装的 Python 完整路径执行 `-m venv .venv-mi`。

启动器固定查找仓库下的 `.venv-mi/Scripts/python.exe`，并自动设置源码路径。上述命令直接调用环境内的解释器，启动控制台和采集任务无需激活 PowerShell 环境，也无需安装 PyTorch、CUDA、Unity、公开 EEG 数据集或预训练模型。

检查 Python 组件：

```powershell
.\.venv-mi\Scripts\python.exe -c "import numpy, scipy, sklearn, matplotlib; from brainflow.board_shim import BoardShim, BoardIds; import pylsl; print('EEG channels:', len(BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_WIFI_BOARD.value))); print('LSL:', pylsl.library_version())"
.\.venv-mi\Scripts\python.exe -m tkinter
```

第一条命令应输出 `EEG channels: 16` 和 LSL 版本；第二条命令应打开 Tk 测试窗口，检查后关闭。后续中文任务提示使用这个窗口组件。

## 3. 安装接收器驱动并建立设备网络

### 配套 USB 路由器：STA 连接

1. 从厂商资料包解压 Windows 驱动，运行 `SR9900_SFX.EXE` 并完成安装。厂商另附 Windows 10 驱动包，可按对应系统使用。驱动安装属于电脑上的独立步骤，`pip install brainflow` 安装的是 Python 采集库。
2. 插入配套 USB 路由器，在设备管理器的“网络适配器”中确认出现 `Corechip SR9900 USB2.0 to Fast Ethernet Adapter`，且没有设备错误标记。如果显示为未知设备，先完成该驱动安装。
3. 打开设备。采用 STA 出厂配置的设备会连接接收器发出的 `LK-USB-Mini-EEG16` 网络，厂商默认密码为 `12345678`；设备默认 IP 为 `192.168.4.1`。STA 蓝灯慢闪表示已连接路由器、等待采集软件。
4. 在 PowerShell 运行 `ipconfig`，找到该 USB 网卡。项目现有接收器给电脑分配的地址为 `192.168.4.2`、网关为 `192.168.4.254`；以当前接收器实际分配结果为准。检查该接口已取得设备子网内的 IPv4 地址。电脑可通过另一块网卡同时访问互联网。

### 设备出厂为 AP：电脑 Wi-Fi 直连

厂商手册给没有配套 USB 路由器的 AP 版本提供另一条路径：打开设备，在电脑 Wi-Fi 列表连接 `LK-Mini-EEG16`，默认密码为 `12345678`。AP 连接成功后绿灯慢闪。采用此方式需要电脑具备 Wi-Fi 网卡。

按设备批次和当前模式选择连接方式。STA 设备没有出现 `LK-Mini-EEG16` 热点时，按配套 USB 路由器流程连接；模式切换方法向厂商取得。

### 检查控制接口与回传端口

```powershell
Test-NetConnection 192.168.4.1 -Port 80
Invoke-RestMethod -Uri http://192.168.4.1/board -TimeoutSec 5
```

第一条命令检查设备 HTTP 控制端口，第二条读取板卡信息。两项可达后再启动采集。失败时检查 USB 网卡驱动、设备供电、当前网络模式和设备 IP。

设备控制请求发往 `192.168.4.1:80`；数据传输由电脑监听 TCP `12345`，设备主动连接电脑回传 EEG。Windows 为此 Python 解释器弹出防火墙提示时，允许设备所接网络上的通信；系统受管时，请管理员为 `.venv-mi/Scripts/python.exe` 配置设备网络的 TCP 入站规则。连接超时排查应同时检查设备 HTTP 和电脑 TCP 回传这两个方向。

## 4. 使用上位机完成第一次设备检查

1. 解压厂商配套的 OpenBCI GUI，运行 `OpenBCI_GUI.exe`。
2. 选择 Cyton＋Daisy、Wi-Fi、16 通道、静态 IP `192.168.4.1`、采样率 `250 Hz`，启动会话。
3. 在 Hardware Settings 中核对启用的 16 路为 Normal 输入、增益 x24、BIAS 开、SRB2 开、SRB1 关。电极接线及异常波形处理见[实验手册](LK-Mini-EEG16运动想象实验操作手册.md#4-电极位置与接线)。
4. 查看波形；需要电极阻抗检查时，使用 GUI 对应组件，操作见[阻抗检查](LK-Mini-EEG16运动想象实验操作手册.md#63-阻抗检查)。项目网页当前提供波形质量筛查，阻抗测量尚未接入。
5. 结束 GUI 数据流并退出设备会话，再使用本项目的 BrainFlow 直连。选择 LSL 路径时，按[实验手册第 7.2 节](LK-Mini-EEG16运动想象实验操作手册.md)配置 GUI 转发，网页选择 LSL 数据源。

## 5. 启动网页并完成采集

在仓库根目录双击 `bci-dashboard.cmd`，或运行：

```powershell
.\bci-mi.cmd ui
```

浏览器地址为 [http://127.0.0.1:8765](http://127.0.0.1:8765)。端口被占用时可运行 `.\bci-mi.cmd ui --port 8766`。网页静态文件包含在仓库中，由 Python 提供，无需安装 Node.js。

先选择“仿真信号”并点击“开始监测”，确认能够显示 16 路波形。停止监测并等待状态变为“已停止”。

真实采集步骤：

1. 完成接线与佩戴检查，设置设备佩戴状态。数据来源选择“真实设备 · Wi-Fi 直连”，设备 IP 为 `192.168.4.1`，端口为 `12345`，采样率为 `250 Hz`。
2. 点击“开始监测”，确认样本数持续增长、16 路波形更新，并按逐路质量结果调整电极。默认通道顺序为 `FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2`，对应 CH1–CH16；参考与 BIAS 使用独立接口。
3. 点击“停止监测”，等待“已停止”，在“03 / 实验流程”选择“采集个人校准数据”，填写受试者编号及会话名称，选择短轮并启动采集。
4. 在弹出的中文提示窗口完成任务。短轮共 20 个计划试次，左右手各 10 次、2 组，提示流程约 3 分 45 秒；断流恢复会增加实际时长。
5. 结束后查看任务日志和“输出位置”，确认保存路径、完整试次数及丢弃数。网页停止任务会请求保存已完成的试次，等待任务退出后再启动下一轮。

需要逐试次独立静息基线时，在“05 / 子实验”选择“运动想象分析”，填写受试者、会话、试次数和分组数，点击“开始提示采集”。真实握拳使用同一区域的“真实握拳对照”入口。两者的数据内容和报告说明见[子实验文档](EXPERIMENTS.md)。

只运行“开始监测”用于查看信号。正式保存任务记录需启动上述采集入口。连续设备检查记录也可通过命令保存：

```powershell
.\bci-mi.cmd check --backend brainflow --duration 60 --output data/recordings/S01/device_check.npz
```

## 6. 找到并复核记录

网页采集保存到 `data/recordings/<受试者编号>/`，文件名带会话及独立后缀。NPZ 中包含连续 EEG、采样时间轴、通道名、采样率、完整任务片段、标签、试次信息与连接分段信息。4 秒任务在 250 Hz 下为每试次 `16 × 1000` 点；子实验另存每试次 `16 × 500` 点的 2 秒静息基线。

用实际文件路径替换下面的示例路径，检查保存内容：

```powershell
.\.venv-mi\Scripts\python.exe -c "import numpy as np; p=np.load(r'data/recordings/S01/实际记录文件.npz',allow_pickle=False); print(p.files); print('epochs:',p['epochs'].shape); print('channels:',p['channel_names']); print('sample_rate:',p['sample_rate'])"
```

采集记录与个体模型目录已被 Git 忽略。克隆仓库后，网页的记录与模型列表初始为空，采集成功后刷新文件列表即可选择新记录。SE-MHAF 个人微调还需另外配置 `.venv`、PyTorch 和预训练模型，步骤见 [SE-MHAF 流程](SE_MHAF_LIVE.md)。

## 7. 软件自检和设备验收

```powershell
.\bci-mi.cmd validate --output-dir outputs/fresh_pc_workflow
```

该命令生成合成数据并执行通道检查、带标签采集、CSP＋LDA 训练和独立回放，结果写入 `outputs/fresh_pc_workflow/workflow_report.json`。

2026-11-14 对 GitHub main `8feb6bb` 的干净源码副本进行了新环境检查：使用 Python 3.12.7 创建 `.venv-mi`，按 `requirements-mi.txt` 安装后依赖检查通过，BrainFlow/LSL 动态库与 Tk 可加载；144 项采集相关测试通过，合成四阶段自检通过。浏览器完成 16 路监测及提示采集中途停止，保存并重载 3 个完整试次，任务数组为 `3 × 16 × 1000`。该次检查没有连接真实设备。

设备验收在新电脑上运行真实数据监测和一轮正式采集，记录接收时长、完整试次数与断流情况。仓库已有 2026-10-31 的设备记录：300 秒未佩戴连接测试发生 11 次自然断流，见[控制台实测记录](CONTROL_CONSOLE.md#软件验证)。main `8feb6bb` 遇到包号不连续会重建连接，采集会丢弃跨连接段的试次；持续传输结果应随新电脑的采集记录核对。
