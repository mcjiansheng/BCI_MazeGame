# LK-Mini-EEG16 运动想象实验操作手册

当前设备配置为 16 通道。SE-MHAF 公开预训练、短轮采集、个人微调及 Unity 控制见 [SE-MHAF 真机流程](SE_MHAF_LIVE.md)。本手册保留接线、实验操作和 CSP＋LDA 对照方法。

本机已配置独立的 `.venv-mi` 环境，直接使用项目根目录的 `bci-mi.cmd` 运行。入口和实测结果见 [本机运行说明](LOCAL_MI_SETUP.md)。下面安装段用于其他电脑；本机不必重新安装原来的 `.venv`。

适用对象：第一次接触 EEG/脑机接口设备的小组成员

目标：使用 LK-Mini-EEG16 的全部 16 个 EEG 通道，采集左右手运动想象，训练一名被试自己的 CSP+LDA 模型，并在 Python 中实时分类。

本手册是当前设备操作入口。设备连接依据仓库厂商通信协议；下面的头皮布局是本项目的建议接线约定，不是设备自动识别出的电极位置。旧文档中的“首轮 8 通道”流程已由本版本替代。16 个通道代表 16 个同步 EEG 测量位置，当前模型仍为左右手二分类。

## 1. 先理解这套系统在做什么

运动想象（Motor Imagery, MI）不是读取“想法内容”。被试在不真正运动的情况下，持续想象左手或右手动作，会使感觉运动皮层附近的 μ 节律（约 8-13 Hz）和 β 节律（约 13-30 Hz）发生统计变化。不同人的频段、空间分布和可识别程度差异很大，因此必须为每名被试进行校准。

完整流程是：

```text
戴帽和接线
  -> OpenBCI GUI 阻抗/波形检查
  -> Python 设备检查
  -> 视觉提示下采集左右手 MI
  -> CSP 提取空间特征 + LDA 分类
  -> 交叉验证和生理特征检查
  -> 加载该被试模型进行实时滑窗分类
```

公开 BCIC2a/2b 模型不能直接当作 LK-Mini 真实用户模型。通道、参考、被试和实验范式都不同。

## 2. 安全与实验边界

1. 本系统是非医疗研究原型，不能用于诊断、治疗或判断被试健康状况。
2. 设备、线束和电极必须按厂商手册使用，不改造供电和模拟前端。
3. 被试有头皮破损、明显不适或无法舒适佩戴时停止实验。
4. 准备电极时不能为了降低阻抗而用力刮伤头皮；被试出现疼痛立即停止。
5. 实验中不使用快速闪烁刺激。被试头晕、恶心、焦虑或疲劳时立即暂停。
6. 原始 EEG、姓名、联系方式和个体模型属于敏感研究数据，不上传公共 GitHub。
7. 使用匿名编号，例如 `S01`、`S02`，不要把姓名写入文件名或程序参数。

## 3. 软件与文件准备

### 3.1 推荐电脑环境

- Windows 10/11 64 位；
- Python 3.11 64 位；
- LK-Mini-EEG16 厂商提供的 OpenBCI GUI v6.0.0 beta 或已验证版本；
- 首次运行允许 OpenBCI GUI/Python 通过“专用网络”防火墙；
- 关闭可能抢占设备端口或网络连接的其他采集程序。

### 3.2 安装 Python 环境

在 PowerShell 中进入仓库根目录：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

如果 PowerShell 禁止执行 `Activate.ps1`，不需要修改系统执行策略：直接用 `.\.venv\Scripts\python.exe -m pip ...` 安装，并把后续命令中的 `python` 换为 `.\.venv\Scripts\python.exe`。若没有 `py`，先安装 Python 3.11，并在安装时勾选 Tcl/Tk 支持；可运行 `python -m tkinter` 验证提示窗口组件。

说明：

- `brainflow`：Python 直接连接 LK-Mini；
- `pylsl`：接收 OpenBCI GUI 转发的 LSL 数据；
- `numpy/scipy/scikit-learn`：预处理与 CSP+LDA；
- `matplotlib`：实时波形和概率显示；
- `-e .`：让 `scripts/` 能导入 `src/bci_maze`。

验证安装：

```powershell
python -c "from brainflow.board_shim import BoardIds; print(BoardIds.CYTON_DAISY_WIFI_BOARD)"
python -c "import pylsl, numpy, scipy, sklearn; print('dependencies ok')"
```

## 4. 电极位置与接线

### 4.1 默认 16 通道接线表

按国际 10-10 扩展电极命名，将设备 CH1-CH16 接到下列头皮位置。`C` 表示中央区，`FC` 为额-中央区，`CP` 为中央-顶叶区；奇数通常位于左半球，偶数位于右半球，`z` 位于中线。C3/C4 及周围是左右手 MI 的主要观察区域。

| LK-Mini 通道 | 10-20 位置 | 作用区域 |
| ---: | --- | --- |
| CH1 | FC3 | 原 core-8，左侧额-中央区 |
| CH2 | FC4 | 原 core-8，右侧额-中央区 |
| CH3 | C3 | 原 core-8，右手 MI 重点观察位置 |
| CH4 | Cz | 原 core-8，中央中线 |
| CH5 | C4 | 原 core-8，左手 MI 重点观察位置 |
| CH6 | CP3 | 原 core-8，左侧中央-顶叶区 |
| CH7 | CPz | 原 core-8，中央-顶叶中线 |
| CH8 | CP4 | 原 core-8，右侧中央-顶叶区 |
| CH9 | Fz | 新增，额中线辅助 |
| CH10 | FC1 | 新增，左近中线额-中央区 |
| CH11 | FCz | 新增，中央额区 |
| CH12 | FC2 | 新增，右近中线额-中央区 |
| CH13 | C5 | 新增，左外侧中央区 |
| CH14 | C6 | 新增，右外侧中央区 |
| CH15 | CP1 | 新增，左近中线中央-顶叶区 |
| CH16 | CP2 | 新增，右近中线中央-顶叶区 |

完整顺序为 `FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2`，与小组提供的布局图一致。该顺序会写入记录文件和模型。若接线与此表不同，必须同时修改 `--channels` 和 `--channel-names`，并让检查、采集和在线分类三处完全一致。模型不会自动推断或重排电极。

这些“口”是头套上的电极位点，不是连接脑内的接口。先按头套印字核对位置，再沿线核对设备插口；不能仅凭电极线颜色判断。若头套没有 FC1/FC2、C5/C6 等位置，应按实际位置制定新映射，不可把别处的电极改名冒充。

### 4.2 参考与 BIAS

除了 16 个 EEG 采集电极，还需要共同参考和 BIAS，参考/BIAS 不计入 16 通道：

- `SRB2`：共同参考，建议接左耳垂 A1；
- `BIAS`：共模噪声抑制，建议接右耳垂 A2；
- EEG 通道使用 Normal 输入，BIAS 打开、SRB2 打开、SRB1 关闭。

如果头套线束已经把参考/BIAS 集成到专用接口，按厂商标识接入。不要把 SRB2 或 BIAS 当成 CH1-CH16，也不要凭线的颜色猜接口。新提供的 `LK-Mini-EEG16/LK-Mini-EEG16 使用说明书 20260716.pdf` 第 4 页明确标注 CH1-CH8 使用 N 输入端，SRB2 接帽上参考 REF（原文写 RFE），BIAS 接帽上 GND。该页图只明确画出前八路，CH9-CH16 和两组参考的线束关系仍应按实物标识或厂家确认，不自行短接未知插口。

该连接方式与 [OpenBCI EEG setup](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EEGSetup/) 和 [OpenBCI ExG setup](https://docs.openbci.com/GettingStarted/Biosensing-Setups/ExGSetup/) 的共同参考方案一致。

### 4.3 从旧 8 通道实验迁移

升级后的默认值会启用全部 16 路。**必须重新采集并训练 16 通道模型**；不能给旧数据补零、复制通道，或只修改旧模型的通道名称。

旧布局 CH1-CH8=`FC3,FC4,C3,Cz,C4,CP3,CPz,CP4` 原封不动保留；只需按 4.1 表把新增八个位置接到 CH9-CH16，不要重排前八根线。虽然 core-8 物理通道保持不变，十六通道 CSP 模型仍需重新采集训练。保留原始八通道记录和模型，给新实验使用独立 session 和模型路径。前一版以 Fz 开头的十六通道模型与当前顺序不兼容，不能只修改标签后加载。

如果要在 LK-Mini-EEG16 上复现旧八电极布局，检查、采集和在线命令都显式附加：

```powershell
--channels 1,2,3,4,5,6,7,8 --channel-names FC3,FC4,C3,Cz,C4,CP3,CPz,CP4
```

这只用于八通道兼容实验，会关闭后八路。回放旧 NPZ 时读取文件实际通道元数据，八通道文件仍可训练八通道模型；八与十六通道会话不能混合训练。旧 `test_mini_eeg_connection.py --port COMx` 是 Mini-EEG08 串口诊断脚本，不适用于本设备的 Wi-Fi 流程。

## 5. 给被试戴帽

### 5.1 实验前

提前告知被试：

- 当天头发保持干净、干燥，尽量不使用发蜡、发油或大量护发产品；
- 实验约 15-25 分钟，中间分 block 休息；
- 任务只需要想象动作，不允许真的动手、动脚或咬牙；
- 可随时要求暂停或退出。

### 5.2 定位头套

1. 让被试坐直，背部有支撑，双脚自然放平，双手放在腿上。
2. 找到鼻根点 nasion、枕外隆凸 inion 和左右耳前点。
3. 将头套中线对准头部中线，Cz 位于前后和左右中点附近。
4. 确认 C3/C4 位于中央区左右两侧，而不是滑到额头或后脑。
5. 头套应稳定但不能造成明显压痛。尺寸不合适时不要强拉，应更换尺寸或调整固定方式。

### 5.3 建立电极接触

1. 分开电极孔下方的头发，让电极尽可能接触头皮而不是压在头发上。
2. 按头套/电极要求加入适量导电膏或导电液。
3. 先处理 SRB2 和 BIAS，再逐个处理 C3、C4、Cz 等关键通道。
4. 整理线缆，固定在椅背或头套后方，避免线缆悬空摆动。
5. 不要让相邻孔中过量导电膏连成一片，以免形成盐桥/短路。

## 6. OpenBCI GUI 中的第一次连接

### 6.1 连接设备网络

本机使用厂商 USB 路由器的 STA 方式。依据 `LK-Mini-EEG16/LK-Mini-EEG16 使用说明书 20260716.pdf` 第 7 页：

1. 插入厂商 USB 接收器；Windows 设备名为 `Corechip SR9900 USB2.0 to Fast Ethernet Adapter`，驱动安装包为 `SR9900_SFX.EXE`。本机已经识别该适配器，无需重复安装。
2. 打开 LK-Mini。设备自动连接接收器提供的 `LK-USB-Mini-EEG16` 网络，默认密码 `12345678`，设备固定 IP 为 `192.168.4.1`。
3. 本机设备侧地址为 `192.168.4.2`，网关为 `192.168.4.254`。内置无线网卡同时连接校园网属于正常情况，不必切断校园网。
4. 蓝灯慢闪表示已连接路由器但未连接采集软件，常亮表示已连接软件。

仅在选择说明书第 8 页的 AP 直连方式时，执行以下步骤：

1. 打开 LK-Mini-EEG16。
2. 在 Windows Wi-Fi 中连接 SSID `LK-Mini-EEG16`。
3. 厂商默认密码为 `12345678`。
4. 连接后电脑可能提示“无 Internet”，这是设备 AP 模式的正常现象。

### 6.2 OpenBCI GUI 会话配置

不同 GUI 版本的按钮名称可能略有差异，但参数应保持一致：

1. Data Source 选择 Cyton+Daisy/Wi-Fi 对应选项；
2. 通道数选择 16；
3. 网络方式选择静态 IP；
4. 设备 IP 填 `192.168.4.1`；
5. 采样率选择 `250 Hz`；
6. 启动会话后打开 Time Series；
7. Hardware Settings 中实际使用通道设置为：Normal、BIAS On、SRB2 On、SRB1 Off；
8. CH1-CH16 全部开启，按 4.1 表核对标签；
9. 初始 PGA Gain 使用 x24。若接触已确认正常但仍 Railed，再尝试 x12 或 x8。

注意：厂商通信协议的表格和示例有一处文字矛盾，但厂商 Python 示例与命令表均使用 `~6` 设置 250 Hz，本项目也采用该命令。

不要将普通 Cyton+Daisy 串口的采样率经验直接套用到本 Wi-Fi 设备。这里 250 Hz 指每路每秒 250 点，16 路的一个 4 秒 trial 是 `16 × 1000`，不是把 250 除以 16。最终还应通过 Python 的时间戳检查确认实测速率。

### 6.3 阻抗检查

打开 Cyton Signal/Impedance Widget：

1. 停止普通数据流后进入阻抗模式；
2. 使用 Check All，或每次只检查一个通道；
3. 31.5 Hz 测试信号会污染附近通道，所以不要同时手动打开多个通道测试；
4. 调整接触后重新检查；
5. 完成后关闭阻抗测试并 Reset Channels，再进入普通采集。

对于湿电极，能够稳定做到 5-10 kΩ 通常较理想；但 Cyton/ADS1299、干电极和不同 GUI 版本的显示阈值并不等价，因此本项目不以一个固定阻抗数字作为唯一标准。优先级是：

1. GUI 显示为 green/yellow 且各通道相对均衡；
2. 普通数据流中没有 Railed、平线或持续巨大工频干扰；
3. Python 质量检查不报告 `bad`；
4. 线缆轻微静止后基线稳定。

若全部通道都同时显示很大阻抗，先检查 SRB2、BIAS 和共同参考。若只有 CH9-CH16 异常，重点检查后八路线束、通道使能和该组参考连接。

## 7. 两种 Python 接入方式

### 7.1 方式 A：BrainFlow 直接连接设备（正式采集推荐）

必须先在 OpenBCI GUI 中停止系统并退出设备会话，避免两个程序抢占连接。

```powershell
python scripts/lk_mini_check.py `
  --backend brainflow `
  --ip-address 192.168.4.1 --ip-port 12345 `
  --sample-rate 250 --gain 24 `
  --channels 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 `
  --channel-names FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2 `
  --duration 60 `
  --output data/recordings/S01/device_check.npz
```

程序会：

- 通过 BrainFlow 的 Cyton+Daisy Wi-Fi 板卡连接；
- 发送 `~6` 设置 250 Hz；
- 启用 CH1-CH16；
- 分两列实时显示 16 条波形，逐路标注设备 CH、电极名与 good/warning/bad；
- 波形仅在显示时去中位数并错开，保存的原始 EEG 不受影响；
- 每 2 秒输出 RMS、峰峰值、平线比例和 50 Hz 干扰比例；
- 结束时报告实际收到的样本数和有效采样率，偏离目标超过 15% 时返回失败；
- 正常退出时释放设备会话。

上述通道和名称就是默认值，按标准接线后可省略两项。检查窗口关闭或按 Ctrl+C 会结束检查并保存已收到的数据。终端还会检查时间戳是否递增、有无采样间隙；质量或采样检查失败时返回非零退出码。终端 `good` 只代表未触发当前信号阈值，不能证明物理接线位置正确，也不能证明已有可识别的 MI。

### 7.2 方式 B：OpenBCI GUI 通过 LSL 转发（联调推荐）

在 OpenBCI GUI 中：

1. 正常连接并启动 LK-Mini；
2. 打开 Time Series Widget；
3. 为避免重复滤波，首轮让 GUI 输出原始 Time Series，由 Python 完成 MI 滤波；
4. 打开 Networking Widget；
5. 选择 LSL、Time Series；
6. Stream Type 使用 `EEG`；
7. Stream Name 例如 `LKMiniEEG`；
8. 启动 LSL stream。

然后运行：

```powershell
python scripts/lk_mini_check.py `
  --backend lsl --lsl-name LKMiniEEG `
  --sample-rate 250 `
  --channels 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 `
  --channel-names FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2 `
  --duration 60
```

如果 GUI 的 LSL 名称不确定，省略 `--lsl-name`，程序会搜索第一个 `type=EEG` 的流。实验室同时有多个 LSL EEG 流时必须指定名称。

LSL 必须输出全部 16 路 EEG，单位为 µV，顺序与接线表一致，采样率为 250 Hz。代码检查流的采样率和通道数量，但不会根据 LSL 标签自动改名，也不自动将 V 换算为 µV。如果 GUI 输出包含 AUX/加速度计，确认选择的是 16 个 EEG 列。这里只配置 Python 订阅端，硬件使能和增益仍由 GUI 设置。

## 8. 如何确认通道真的接对了

Python 只能检测平线、幅值异常和噪声，不能从一小段 EEG 自动知道“CH3 的线是否真的插在 C3”。需要人工完成映射验证：

1. 打开实时波形；
2. 先按 4.1 的 16 行表逐根追踪电极标签和设备插口，逐项记录；可在厂商允许的前提下依次轻触电极固定件辅助观察；
3. 每次只动一个位置，不拔线，不触碰裸露导体；
4. 对应通道可能出现明显瞬态变化，但共同参考和线缆牵动也会影响其他通道，不能仅凭“哪一路变化最大”判断接线；
5. 若另一个通道变化最大，检查线束标签和 `--channel-names`；
6. 测试结束后等待波形恢复稳定，再采 MI。

附加测试：

- 被试连续眨眼时，Fz 等额区通道通常更明显；不能把眨眼测试当作全部 16 路的定位证据；
- 咬牙会产生广泛高频肌电，只用于确认系统对伪迹有响应，不能作为 EEG 有效性的证据；
- 睁眼/闭眼 α 差异在枕区最明显，本项目没有 O1/O2，所以中央区变化不明显也不代表设备故障。

## 9. 被试应该怎样进行运动想象

采用“动觉想象”，即想象动作本身的肌肉、关节和用力感觉，而不是从第三人称视角看一只手移动。

### 左手任务

- 想象左手持续、重复地握紧和放松；
- 想象手指弯曲、手掌用力和松开的感觉；
- 左手实际保持放松，不发生可见运动。

### 右手任务

- 与左手相同，但集中在右手；
- 不要同时想象左右手；
- 不要通过咬牙、屏住呼吸或肩膀用力“帮助”分类。

### 每个 trial 中应该做什么

| 阶段 | 默认时长 | 被试行为 |
| --- | ---: | --- |
| 准备 | 2 s | 注视 `+`，放松，不提前想象 |
| 任务提示 | 1 s | 看清左右方向，仍保持静止 |
| 开始运动想象 | 4 s | 持续进行对应动觉想象 |
| 休息 | 随机 2-3 s | 停止想象，可以自然眨眼和放松 |

每个 block 结束后额外休息 30 秒，可用 `--block-rest-seconds` 调整。16 通道不要求被试分别“激活 16 个位置”：被试始终只完成屏幕指定的左手或右手动觉想象。

正式采集前做 5-10 次不保存的练习。操作员观察手指、前臂、下颌和肩膀，确认没有实际运动。

## 10. 采集左右手 MI 校准数据

直接连接方式：

```powershell
python scripts/collect_lk_mi.py `
  --subject S01 `
  --backend brainflow `
  --sample-rate 250 --gain 24 `
  --channels 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 `
  --channel-names FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2 `
  --classes left_hand,right_hand `
  --trials-per-class 40 --blocks 4 `
  --fullscreen
```

使用 GUI+LSL 时，把接入参数替换为：

```powershell
--backend lsl --lsl-name LKMiniEEG
```

界面功能：

- 全屏显示准备、箭头、运动想象和休息；
- 显示 block/trial 和剩余时间；
- 底部显示 `good 通道数/16`，并列出 warning/bad 电极名称；等待数据时显示等待状态；
- 操作员发现真实动作、说话、咳嗽或明显线缆扰动时按 `A`，当前 trial 会被标记为伪迹；
- 按 `Esc` 安全提前结束，已完成 trial 仍会保存。

默认输出：

```text
data/recordings/S01/<session>.npz
```

文件包含原始连续 EEG、4 秒 trial、标签、block、伪迹标记、采样率、通道顺序和参数。该目录已被 Git 忽略。

上方命令显式指定完整采集方案，共 80 个 trial，分 4 个 block，每组左右手各 10 次、组内随机。当前省略次数参数时采用 `--preset short`，共 20 次、2 组，约 3 分 45 秒；`standard` 为 40 次、4 组，`full` 为上述完整方案。每次使用新的 session/输出文件名；被试先看方向，等“开始运动想象”后再开始。操作员标记伪迹，后续训练排除已标记试次。信号异常时可按 Esc 结束，改善接触后采集新会话。

无设备熟悉被试界面可运行 `python scripts/collect_lk_mi.py --subject PRACTICE --backend synthetic --trials-per-class 2 --blocks 1`。这会生成仿真记录，不能混入真人 S01 的训练数据。`--simulate-mi` 是跳过界面的自动化验收模式，与被试练习窗口不同。

## 11. 训练该被试的识别模型

```powershell
python scripts/train_lk_mi.py `
  data/recordings/S01/20260831-140000.npz `
  --subject S01
```

多个会话可以一起输入：

```powershell
python scripts/train_lk_mi.py `
  data/recordings/S01/session1.npz `
  data/recordings/S01/session2.npz `
  --subject S01
```

默认：

- 排除按 `A` 标记的 trial；
- 7-30 Hz 带通；
- 4 个正则化 CSP 分量；
- shrinkage LDA；
- 优先按 block 做分层分组交叉验证；
- 另外用时间上最后一个 block 做一次只训练于此前 block 的时序留出验证；
- 输出 Accuracy、Balanced Accuracy、Macro-F1、κ、混淆矩阵和每折结果；
- 输出 8-13 Hz 与 13-30 Hz 各通道的类别效应量；
- 计算当前 trial 数量下二分类机会水平的 95% 上界。

数据维度和处理细节：连续 EEG 为 `16 × 样本数`；默认 trial 为 `16 × 1000`；80 次完整 trial 为 `80 × 16 × 1000`。训练从 NPZ 读取采样率和电极名，无需再指定 `--channels`。带通使用四阶 Butterworth SOS，逐 trial 双向滤波；CSP 在全部 16 路上拟合正则化协方差，再输出默认 4 个空间分量的归一化对数方差，交给 LDA。4 个 CSP 分量不等于只使用 4 个采集电极，不必因通道翻倍就把分量改成 16。

每折只用训练折拟合 CSP/LDA，在线调用保存的同一滤波及分类 pipeline。这里没有自动 ICA、自动插值坏道或独立 50 Hz 陷波；7-30 Hz 带通会抑制带外成分，工频质量指标则在原始信号上计算。明显坏试次应由操作员标记排除，不应靠增加电极数量掩盖伪迹。

报告包含 `channel_count: 16`、有序 `channel_names` 和每个电极的 μ/β 效应量。该效应量表示左右手 trial 的类别差异。最终模型使用全部保留 trial 重新拟合，报告另列分组交叉验证结果。模型和报告保存成功后返回退出码 0；结果未超过统计阈值时，报告保留未通过状态。真实设备或 LSL 运动想象记录训练的个人模型可用于闭环联调，在线日志保留实际校准状态。

默认模型位置：

```text
outputs/subject_models/S01/csp_lda.joblib
outputs/subject_models/S01/csp_lda.json
```

如已有八通道同名模型，训练时使用 `--output outputs/subject_models/S01/csp_lda_16ch.joblib`，在线命令的 `--model` 也指向此文件，以保留旧模型。省略 `--output` 会写入上面的默认路径，已有同名文件会被覆盖。

### 怎样判断“获得了切实有效的 MI 数据”

不能只看一段波形，也不能只看训练集准确率。至少同时满足：

1. 信号检查没有持续平线、Railed 或严重 50 Hz 污染；
2. 左右手 trial 数量平衡，伪迹比例不过高；
3. 分组交叉验证准确率超过报告中的 `chance_accuracy_95_percent`；
4. `temporal_holdout` 的最后 block 结果也超过其对应机会水平，多个 fold 表现一致；
5. C3/C4 及附近通道在 μ/β 频段出现合理的类别差异；
6. 换一个会话再次采集时，模型仍明显高于机会水平。

若准确率没有超过机会水平，不要继续让模型控制迷宫。先检查实际运动、提示时间、通道映射、接触质量、被试策略和疲劳。

## 12. 实时分类

```powershell
python scripts/online_lk_mi.py `
  --model outputs/subject_models/S01/csp_lda.joblib `
  --backend brainflow `
  --sample-rate 250 --gain 24 `
  --channels 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 `
  --channel-names FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2 `
  --step-seconds 0.5 `
  --confidence 0.65 --margin 0.15 `
  --plot
```

程序使用最近 4 秒数据，每 0.5 秒更新一次。默认连续平滑 3 个窗口：

- 概率不足 0.65：`unknown`；
- 第一、第二类别差距不足 0.15：`unknown`；
- 任一路信号为 warning/bad（包括严重工频污染）：`unknown`；
- 模型与输入采样率/通道顺序不一致：直接报错并停止。

日志保存在 `outputs/online/`。所有 `unknown` 都不会发送移动。

概率图标题显示模型通道数和被试编号；坏道名单分行显示，包含 CH9-CH16 对应电极。首次需要收满 4 秒窗，默认 3 窗平滑还需再更新两次，通常约 5 秒后才可能输出首个接受结果。左右手二分类没有训练“静息”类别，因此 **unknown 不是可靠的静息检测**：不做 MI 时也可能输出高置信度左右手。连接游戏前须测量静息误触发率，并由操作员控制实验开始/结束。

可选连接 Unity：

```powershell
python scripts/online_lk_mi.py `
  --model outputs/subject_models/S01/csp_lda.joblib `
  --backend brainflow `
  --channels 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 `
  --channel-names FC3,FC4,C3,Cz,C4,CP3,CPz,CP4,Fz,FC1,FCz,FC2,C5,C6,CP1,CP2 `
  --unity-host 127.0.0.1 --unity-port 7777 `
  --command-cooldown 1.5 --plot
```

当前映射是左手=`left`、右手=`right`。Unity 最终应采用“相对左转/右转或岔路选择”，而不是把它们解释成世界坐标的永远向左/向右。

## 13. 无设备回放和开发

一键运行“通道检查 -> MI 采集文件 -> 预处理/训练 -> 独立在线回放分类”的完整软件验收：

```powershell
python scripts/validate_lk_mini_workflow.py
```

该验收使用显式标记的 16 通道合成 MI 数据，输出位于 `outputs/lk_mini_workflow/`。`workflow_report.json` 必须包含 `status: passed`、`device_contract.channel_count: 16`，且所有 checks 为 true。通过表示四个命令行入口、NPZ 元数据、CSP+LDA 保存回载、在线窗口、EOF 退出和坏道拒识能够协同工作；它不替代真实设备的 Wi-Fi、阻抗、物理通道位置、持续采样率和真人 MI 有效性验收。合成信号的分类差异是人为构造的，不模拟或证明真实脑区生理分布。

检查 UI/脚本：

```powershell
python scripts/lk_mini_check.py --backend synthetic --duration 10
```

回放以前的连续记录：

```powershell
python scripts/lk_mini_check.py `
  --backend replay --replay data/recordings/S01/session.npz `
  --sample-rate 250 --duration 60
```

在线模型也支持 `--backend replay`。必须先通过回放验证训练和在线路径，再连接真人控制 Unity。

## 14. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| BrainFlow 连接超时 | 是否连接设备 Wi-Fi；设备 IP/端口；OpenBCI GUI 是否仍占用连接；防火墙专用网络权限 |
| 找不到 LSL stream | GUI Networking 是否启动；类型是否 EEG；Time Series Widget 是否打开；名称是否与 `--lsl-name` 一致 |
| 16 个通道全部阻抗很大 | 先查 SRB2、BIAS、耳垂接触和公共线束，不要先重做所有头皮电极 |
| 某个通道平线 | 通道是否被关闭；线是否插错；电极是否接触头皮；对应线缆是否损坏 |
| 只有后 8 路平线/缺失 | 是否仍使用旧八通道参数；GUI 是否选 16 路；CH9-CH16 是否启用；后八路线束与参考连接 |
| Railed | 接触不良、参考脱落或增益过高；先改善接触，再将 x24 降到 x12/x8 |
| 50 Hz 很强 | BIAS/参考、线缆摆动、附近电源/充电器、接地环境；不要只靠陷波掩盖问题 |
| 采集 trial 数量不足 | 检查流是否中断；4 秒阶段内是否实际收到 1000 个样本；不要强行补零 |
| 模型提示通道不匹配 | 按模型记录的顺序重新接线/传参；不能通过改文件绕过检查 |
| 交叉验证接近 50% | 先验证接线和真实动作伪迹，再调整 MI 策略；增加数据前先确保范式正确 |
| 实时结果快速跳变 | 增大平滑 history、提高置信度/差值阈值、延长命令冷却；先查看坏道和肌电 |

## 15. 每次正式实验检查表

### 操作员开始前

- [ ] 已获得知情同意并使用匿名 ID；
- [ ] 头套位置正确、被试舒适；
- [ ] CH1-CH16 映射逐路确认并记录，尤其 CH3=C3、CH5=C4，前八路与原实验一致；
- [ ] SRB2 和 BIAS 已确认；
- [ ] 阻抗检查完成并退出阻抗模式；
- [ ] 普通波形无 Railed/平线；
- [ ] Python 16 个通道均收到数据，实测采样率与时间戳检查通过；
- [ ] 被试完成 5-10 个练习 trial；
- [ ] 提醒被试在 4 秒想象期间不动、不说话、不咬牙；
- [ ] 输出目录有足够空间。

### 结束后

- [ ] 确认 NPZ 能重新打开；
- [ ] 记录异常、暂停和被试策略；
- [ ] 清洁头套和电极；
- [ ] 备份到受控存储，不上传公共仓库；
- [ ] 训练报告与对应 session 绑定；
- [ ] 未超过机会水平时不进入游戏控制实验。

## 16. 主要参考

本机实测发现 BrainFlow 的 Wi-Fi 时间戳按网络批量到达，间隔不代表 ADC 采样间隔。直连路径现在先校验拼帧后设备包序号按 0-255 循环递增，再建立等间隔样本时间轴；诊断 NPZ 同时保存原始 `received_timestamps` 用于实测速率检查。真实序号跳变会停止采集/在线分析，不把断流两侧拼成有效窗口。LSL 和回放按提供的样本时间戳检查间断；回放保留原始间隔，不再掩盖旧记录中的缺口。BrainFlow 5.22.2 已根据各通道配置增益返回微伏值，Python 不再额外缩放。先前带 `24/gain` 补偿的 x6 诊断幅值偏大四倍，应重新测量；x24 不受该问题影响。

采集连续 5 秒无数据会报错，已完成试次仍保存。无设备自动生成数据使用 `--simulate-mi`；视觉采集禁止 `--no-realtime`，以免快速生成的样本和屏幕提示不同步。训练发现 NaN/Inf 会报错，不能静默转成零信号。当前二分类训练器只接受两个任务类别。

- [LK-Mini-EEG16 厂商通信协议与示例](../vendor/LK-Mini-EEG16/README.md)
- [BrainFlow Python examples](https://brainflow.readthedocs.io/en/stable/Examples.html)
- [OpenBCI GUI](https://github.com/OpenBCI/OpenBCI_GUI)
- [OpenBCI GUI Networking/LSL](https://docs.openbci.com/Software/OpenBCISoftware/GUIWidgets/#networking)
- [OpenBCI EEG setup](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EEGSetup/)
- [MNE CSP motor imagery example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [MOABB cross-session motor imagery](https://moabb.neurotechx.com/docs/auto_examples/paradigm_examples/plot_cross_session_motor_imagery.html)
- [FBCSP paper](https://doi.org/10.3389/fnins.2012.00039)
