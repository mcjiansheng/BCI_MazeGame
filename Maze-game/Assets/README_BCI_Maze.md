# Unity BCI Maze：3D 第一人称 Demo

Unity 2022.3.57f1c1，打开 `Scenes/SampleScene.unity` 后 Play 即可。默认 15 × 15，运行时生成墙体、步道、阳光、引导灯带、出口和第一人称相机，无需额外美术资源。

## 流程

- 直道自动前进，唯一出口的弯道自动转弯。
- 分岔处停留 4 秒供选择；有直行路则到时自动直行，没有直行路的 T 字口一直等待左右选择。
- 左右指令是相对当前面朝方向的转向，不是世界坐标平移。
- 行进或转身过程中可提前预约，遇到下一个对应方向可走的格子时执行。新指令替换旧预约，不累积一串转向。
- 掉头在下一个格子中心执行，避免在墙边转身；死胡同默认等待 2 秒后自动掉头，可关闭自动掉头并手动发送。
- 无指令时保持自动行走；脑机输出 unknown / 不发送指令不会暂停游戏。
- 到达橙色出口停止，显示步数、转向次数和用时。

## 操作

| 输入 | 行为 |
| --- | --- |
| A / ← | 预约左转 |
| D / → | 预约右转 |
| S / ↓ | 预约掉头 |
| W / ↑ | 清除预约并继续；路口有直行出口时立即直行 |
| Space | 暂停 / 恢复 |
| R | 新地图并重置进度 |
| H | BFS 下一步提示 |
| M / V | 显示 / 隐藏小地图 |

屏幕下方提供左右转、掉头按钮，左下角提供暂停、重建与设置，右下角是提示。设置可调步速、路口等待时长、死胡同自动掉头和小地图。核心参数也直接暴露在 Inspector 中。

## UDP 协议

默认 `127.0.0.1:7777`，兼容纯文本与 JSON：

```json
{"command":"left","ts":123.4}
```

| 指令 | 含义 |
| --- | --- |
| left / turn_left | 相对左转 |
| right / turn_right | 相对右转 |
| uturn / turn_around | 掉头 |
| forward | 清除预约、恢复前进；不能穿墙 |
| pause / stop | 暂停自动导航 |
| resume | 恢复自动导航 |
| cancel | 取消预约，不改变暂停状态 |
| restart | 重新生成迷宫 |
| heartbeat / status | 不移动，返回 ACK 与状态 |

旧 `up/move_up/w/north` 兼容为 forward，`down/move_down/s/south/back/backward` 兼容为 uturn，`a/west/move_left` 和 `d/east/move_right` 分别兼容 left/right。ACK 的 command 是归一化后的名称。

```json
{
  "command": "left",
  "accepted": true,
  "echo_ts": 123.4,
  "navigation": {
    "mode": "first_person_relative",
    "state": "Moving",
    "paused": false,
    "x": 1,
    "y": 2,
    "heading": 0,
    "heading_name": "N",
    "pending_turn": "left"
  }
}
```

示例节选了 navigation 字段。实际还返回步数、转向次数、左右/前后出口、路口剩余等待时间、终点和用时。heading 0/1/2/3 对应北/东/南/西。移动期间 x/y 是最近到达的格子中心；Turning 期间 heading 是本次转身的目标方向。

**accepted=true 表示指令已被接受，可能只是预约成功，不等于已经转身或移动。** 可以用 status 查询 pending_turn 和 state。暂停时也允许预约，恢复后执行；通关后转向返回 false。heartbeat/status 不会触发移动。

UDP 工作线程只接收并入队，主线程执行导航并返回 ACK。`ExternalMazeInput.SubmitCommand` 可从其他线程调用，返回值仅表示入队；`MazeGameController.SubmitExternalCommand` 是主线程同步入口。这样无需在游戏状态上维护并发锁。

## 接入脑机分类

现有左右手二分类的 left/right 可以直接使用。只靠二分类仍可在 T 字口选择方向，在直行分岔时选择转弯或等待直行，死胡同由自动掉头恢复。

```powershell
.\bci-mi.cmd online --model outputs/subject_models/S01/semhaf16.pt --backend brainflow --unity-host 127.0.0.1 --plot
```

默认路口模式取得暂停确认后重新积累 4 秒脑电与平滑历史，每个路口接收一个稳定决策。T 字口为左右选择；单侧出口与直行组合时，左右手对应左右排列的两个选项。三出口处左右手对应左转、右转，直行保留键盘和按钮入口。断流取消旧预约，恢复后重新积累数据。训练和采集入口见 [SE-MHAF 真机流程](../../doc/SE_MHAF_LIVE.md)。

状态回执新增 `maze_generation`，每次生成地图递增。JSON 请求可携带 `expected` 对象，包含 `maze_generation,x,y,heading,steps,paused,state,pending_turn`；Unity 主线程在执行前逐项核对，过期条件返回 `accepted=false`。省略 `expected` 保留原有协议行为。

BCIC2a 回放可选双脚掉头映射：

```powershell
python scripts/run_bci_maze_integration.py --host 127.0.0.1 --port 7777 --label-map configs/label_maps/bcic2a_relative.json
```

左右手分别 left/right，双脚 uturn，舌头不发送指令。该映射不会让现有二分类模型额外识别双脚；主动掉头也可由按键/鼠标完成。

## 代码与验证

- `Scripts/MazeGameController.cs`：DFS、导航状态、平滑移动/转向、BFS 与查询接口。
- `Scripts/MazeWorldRenderer.cs`：3D 几何体、共享材质和照明。
- `Scripts/MazeHud.cs`：HUD、小地图、按钮与设置。
- `Scripts/ExternalMazeInput.cs`：UDP 队列、结构化 JSON、ACK。

在仓库根目录、Unity Play 模式下运行：

```powershell
uv run scripts/validate_unity_maze.py
```

验证会重建当前运行中的地图，以加速步速检查实际 UDP 通关，结束后恢复原步速/等待参数并暂停在新地图起点。它不使用 EEG 数据，也不代表真实脑机通关率。设计与验证结果见 `doc/UNITY_FIRST_PERSON_DEMO.md`；MCP 配置见 `doc/UNITY_MCP_SETUP.md`。
