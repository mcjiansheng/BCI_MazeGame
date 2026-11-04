# LK-Mini 连接恢复与传输验证

BrainFlow 采集入口默认启用自动重连。检查、提示采集和在线分类使用同一数据源，设备恢复后继续收数，并用段编号区分各次有效数据流。

## 会话恢复流程

采集库在独立工作进程中执行连接、设备配置和读数。主程序持续处理窗口操作与任务期限。单次连接超过 12 秒、连续 3 秒没有有效样本，或包序号检查发现跳变时，程序进入恢复状态，先请求旧工作进程停止并释放会话。清理等待超过 4 秒时终止该工作进程，确认退出后重新建立连接。

每次建立会话都会重新设置采样率、通道和增益。收到恢复后的首个有效数据块才计为成功重连。连续连接失败采用递增的重试间隔；设备重新可达后自动继续。参数错误或会话配置冲突会结束任务并报告原因。停止操作会取消后续重试并清理当前工作进程。

控制请求发送至设备 `192.168.4.1` 的 HTTP 端口；电脑监听 TCP `12345`，由设备主动连接并发送数据。一次运行使用一个采集入口，启动前结束其他占用该设备的采集软件。

## 日常入口参数

下列参数适用于 `bci-mi.cmd check`、`collect`、`online` 的 BrainFlow 后端。

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--auto-reconnect` / `--no-auto-reconnect` | 开启 | 选择自动恢复或单次会话 |
| `--no-data-timeout` | 3 秒 | 持续无有效样本时重新建立会话 |
| `--connect-timeout` | 12 秒 | 单次工作进程建立会话的期限 |
| `--device-timeout` | 5 秒 | BrainFlow 单次 HTTP 控制请求的超时 |
| `--retry-delay` | 0.5 秒 | 初始重试间隔 |
| `--max-retry-delay` | 3 秒 | 递增重试间隔的上限 |
| `--max-retries` | 0 | 0 持续重试至用户停止；正数限制连续失败后的重试次数 |
| `--shutdown-timeout` | 4 秒 | 请求旧工作进程正常退出的等待期限 |

```powershell
# 使用默认恢复策略并显示十六路波形
.\bci-mi.cmd check --backend brainflow --duration 60 --output outputs/lk_check/capture.npz

# 限制连续失败后的重试次数
.\bci-mi.cmd check --backend brainflow --duration 60 --max-retries 3

# 用个人模型恢复在线分类
.\bci-mi.cmd online --model outputs/subject_models/S01/core8_16ch.joblib --backend brainflow --plot
```

## 数据分段与实验处理

数据源检测到连接中断时立即递增 `segment_id`。同一段内部按设备包序号验证连续性，并按采样率建立样本时间轴；恢复后的段重新定位时间起点，保留停顿形成的时间间隙。原始 Wi-Fi 接收时间单独保存，供吞吐和批量到达分析使用。

- **通道检查**：清空当前显示和质量窗口，继续保存已经验证的数据；NPZ 包含逐样本段编号、各段范围、恢复次数和事件。
- **提示采集**：断流后等待新段具备完整信号再开始下一次提示。跨段或样本不足的当前试次写入丢弃记录，保留此前完成的试次及连续数据。任务结束时输出有效试次数和丢弃数。
- **在线分类**：段变化时清空滑窗、概率平滑历史及发送状态，记录 `unknown`。新段达到模型要求的完整窗口和概率历史后恢复判决。
- **记录回放**：读取保存的段编号，并在段界截断数据块，使检查、录制和在线模块按同一规则处理恢复记录。

## 实机传输验证

[验证脚本](../scripts/validate_lk_connection.py)记录指定时长内的实际收数、段内连续性、恢复事件和退出情况。默认配置为 16 通道、250 Hz、增益 24，自动恢复开启。`--wear-state` 记录佩戴条件。

```powershell
# 未佩戴：持续传输观察
.\.venv-mi\Scripts\python.exe scripts/validate_lk_connection.py --duration 300 --wear-state unworn --output-dir outputs/lk_connection/continuous

# 主动重建两次会话，检查恢复与分段
.\.venv-mi\Scripts\python.exe scripts/validate_lk_connection.py --duration 120 --restart-at 30,70 --wear-state unworn --output-dir outputs/lk_connection/restart

# 通过设备控制接口停止两次数据流，检查无数据检测和自动恢复
.\.venv-mi\Scripts\python.exe scripts/validate_lk_connection.py --duration 120 --stop-stream-at 20,70 --wear-state unworn --output-dir outputs/lk_connection/stream_stop

# 单次会话对照
.\.venv-mi\Scripts\python.exe scripts/validate_lk_connection.py --duration 60 --single-session --wear-state unworn --output-dir outputs/lk_connection/single_session
```

每次验证输出 `connection_report.json`；收到数据时同时保存 `capture.npz`。报告包括连接等待时长、总样本数、整段吞吐、最长收数间隔、尾部无数据时长、各段连续性及恢复事件。主动重建记录为 `forced_reconnects`，自然故障记录为 `failures`，已有数据流的自然中断记录为 `gaps`。

持续传输验收检查整段完成、有限数值、段内及跨段连续性、首包等待不超过 15 秒、预期有效采集时段的样本覆盖率至少 90%，以及末尾无数据时间不超过 1 秒。恢复验收逐次核对注入成功、断流与恢复事件、新段至少 1 秒有效样本及成功重连计数；计划注入漏执行也会导致验收失败。脚本退出码依据完整 `validation_passed` 结果，详细失败原因保存在 `validation_failures`。新报告保存统一单调时钟原点及各段数据交付时间。

未佩戴测试评价设备传输与会话恢复。佩戴信号质量采用通道检查记录；带标签运动想象数据和个人分类模型按[实验手册](LK-Mini-EEG16运动想象实验操作手册.md)另行采集与评价。

## 当前验证状态

Anaconda 2024.10-1 与 Python 3.12.7 已恢复至 `C:\Users\48811\anaconda3`，两个项目环境的依赖检查均通过。研究环境已恢复 PyTorch 2.5.1+cu121，并在 RTX 4080 Laptop GPU 上完成张量计算检查。完整回归套件 105 项通过，包含重试、超时、真实工作进程取消、数据分段及消费者处理；测试记录为 `outputs/connection_20261029/pytest.xml`。合成采集、训练和在线回放流程验收通过，报告为 `outputs/environment_restore_20261029/workflow_report.json`。

本次实机验证采用未佩戴条件，设备为原 LK-Mini-EEG16，16 通道、250 Hz、增益 24。测试结果如下：

| 验证 | 结果 | 记录 |
| --- | --- | --- |
| 300 秒持续传输 | 收到 16 × 74388 点；包含 2.484 秒首次连接等待的整段吞吐为 247.96 点/秒；最长收数间隔 0.109 秒；断流 0 次，段内连续性错误 0 次 | `outputs/connection_20261029/auto_recovery/connection_report.json` |
| 120 秒主动停流 | 第 20、70 秒通过设备 `/stream/stop` 停止发送，两次均在无数据 3 秒后自动重连；恢复后持续收数，共 25200 点，分为 3 段，段内连续性错误 0 次 | `outputs/connection_20261029/stream_stop_recovery/connection_report.json` |
| 恢复耗时 | 依据原始接收时间，最后一帧至下一段首帧间隔分别为 8.482、8.499 秒；从程序检测到断流至恢复首个有效块分别为 5.468、5.500 秒 | 同目录 `capture.npz` 与恢复事件 |
| 日常启动器 | `bci-mi.cmd check` 完成 30 秒检查并保存 6744 点；有效收数阶段 250.3 Hz，断流 0 次。未佩戴时 16 路质量状态均为 bad，数据保留为传输检查记录 | `outputs/connection_20261029/launcher_check.npz` |

这两次停流由设备控制接口触发；恢复由默认无数据检测执行，未调用程序的主动重建接口。早期单次会话和原始 TCP 对照曾出现停顿，随后用户确认采集盒熄灭并重新开机。上表持续传输与恢复验证在重新开机后完成。设备断电期间程序持续重试，设备重新可达后继续建立会话。

[厂商通信协议](../vendor/LK-Mini-EEG16/LK-Mini-EEG16-通信协议-20260814.pdf)第 5 页规定，同一采样序号的两个 33 字节帧合成 16 路。实测原始帧为 `n,n,n+1,n+1`，符合该模式；设备 `/board` 控制接口同时返回 `cyton` 和 8 路状态字段。通道模式按厂商协议、实际帧对和 16 路解码结果核对。

历史设备通信与佩戴质量记录见 [2026-09-05 断流诊断](LK_TRANSPORT_DIAGNOSIS_20260905.md)。

## SE-MHAF 真机复验

采集盒重新开机后完成 180 秒连续采集和两轮各 120 秒的停流恢复测试，配置仍为 16 通道、250 Hz，佩戴条件为未佩戴。连续采集收到 44424 点/通道，最长收数间隔为 0.094 秒，意外断流和连续性错误均为 0。两轮合计 4 次设备 HTTP 停流均由默认 3 秒无数据检测触发自动恢复。

正式 SE-MHAF 在线入口的两次停流至恢复收数耗时为 9.547、9.266 秒，恢复后重新形成 4 秒新窗口；175 个窗口被质量门控判为 `bad_signal`，接受决策数为 0。验收脚本及完整记录见 [SE-MHAF 实机流程](SE_MHAF_LIVE.md)。

该轮在线测试正常退出后的网络复查再次发现采集盒无响应，接收器和主机直连路由保持可用；采集盒供电与无线状态为后续现场核实项。已完成的传输测量、恢复验收及后续失联记录分别保存在 `outputs/semhaf_live_20261029` 下的独立目录。
