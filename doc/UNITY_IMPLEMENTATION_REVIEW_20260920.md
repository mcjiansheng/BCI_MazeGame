# Unity 游戏实现检查（2026-09-20）

> 本文记录改造前的 2D 版本。当前已实现 3D 第一人称、自动前进与相对转向，最新流程及验收见 [3D Demo 说明](UNITY_FIRST_PERSON_DEMO.md)。

## 项目定位

仓库包含 BCIC2a/2b 离线运动想象模型实验、LK-Mini-EEG16 采集与受试者 CSP+LDA 在线分类，以及 Unity 迷宫反馈游戏。Unity 接收的是方向指令，不在内部执行 EEG 模型。当前游戏已达到可运行、可通过 UDP 完整通关的原型阶段；真人脑机自主通关仍需实验与控制策略验证。

## 优先处理的缺口

1. **脑机输出不足以覆盖迷宫所需方向。** `scripts/online_lk_mi.py:27` 将左右手映射为 left/right；当前左右手二分类模型不能输出上下移动。BCIC2a 默认映射也没有 down（舌头不移动）。应先决定交叉口二选一、转向加自动前进或方向扫描等交互策略，再评价脑机通关率。本次通关使用 BFS 生成的指令，不代表 EEG 分类通关。
2. **重建迷宫与网络移动存在竞争条件。** `MazeGameController.cs:151` 的 GenerateMaze 不获取 mazeStateLock，而 `:217` 的外部移动在工作线程持锁访问同一数组、位置和尺寸。UDP 持续发送期间按 R 或改变尺寸重建，可能读到部分初始化状态。建议统一进入主线程处理，或在同一锁下原子发布完整迷宫状态。此项来自代码检查，未做并发压力复现。
3. **重复生成的运行时图形资源未释放。** `MazeGameController.cs:324` 和 `:337` 每次重新生成 Texture2D 与 Sprite，未销毁旧资源；长时间反复 R 有资源增长风险。应释放替换的资源并补充生命周期清理。此项尚未做内存曲线测试。
4. **JSON 接收解析不严格。** `ExternalMazeInput.cs:166` 通过字符串搜索提取字段，没有 JSON 结构及转义校验；非法 JSON 或嵌套字段可能被当作 command。建议改用结构化 JSON 解析并明确非法输入的 ACK 行为。目前合法 JSON 和纯文本是预期协议，不能把宽松解析当作验证。

## 已实现和实测结果

| 项目 | 结果与证据 |
| --- | --- |
| 场景 | SampleScene 已加载、未修改，含游戏对象和主相机 |
| 编译与 Console | 检查时未编译中，无待处理域重载；运行前后查询 error/warning 均为 0 条 |
| 迷宫 | 实际运行生成 61 × 61，起点 (1,1)，终点 (59,59) |
| BFS | 本次随机地图起点到终点最短路径为 544 步 |
| MCP | 初始化、工具/资源发现、项目状态、场景、Console、执行 C#、截图、Play/Stop 均成功 |
| UDP | 按 BFS 顺序发送 544 次真实 UDP 移动，全部 accepted=true |
| 边界与终点 | 起点向左撞墙和终点后继续移动均 accepted=false |
| 心跳与时间戳 | heartbeat ACK 成功，所有测试报文 echo_ts 与发送值一致 |
| 渲染同步 | 最终 Maze Player 的 Transform 为 (59,59,0)，HUD 显示 Maze complete in 544 steps |
| 测试清理 | 已退出 Play，恢复测试前的编辑模式；未保存运行时改动 |

测试期间移动间隔为约 10 ms，是软件链路检查，不能用于推断真人 MI 操作速度。游戏外部 UDP 路径没有键盘冷却限制。

代码中还有 WASD/方向键、R 重建、H 最短路长度、V 视角切换；本次未模拟实际键盘逐项验收。H 当前显示路径长度，不在画面上绘制路线。图形为运行时纹理和 Sprite，HUD 使用固定像素 IMGUI；没有进行多分辨率、独立构建或长时间稳定性测试。

## 本地证据

- `outputs/unity_mcp/check.json`：运行前 MCP 连接、实例、场景与控制台。
- `outputs/unity_mcp/tools.json`：实际发现的 MCP 工具 schema。
- `outputs/unity_mcp/runtime.json`：当前迷宫尺寸及完整 BFS 路径。
- `outputs/unity_mcp/udp.json`：544 次移动以及心跳、拒绝动作的 ACK。
- `outputs/unity_mcp/completed.json`：终点位置、剩余路径与 UDP 处理计数。
- `outputs/unity_mcp/play-check.json`：运行结束、停止 Play 之前的状态与 Console。
- `Maze-game/Library/MCPForUnity/Verification/bci-maze-follow.png`：起点相机画面。
- `Maze-game/Library/MCPForUnity/Verification/bci-maze-complete.png`：带 HUD 的终点画面。

以上是本地生成证据，不是必须提交的源码；Library 会在清理 Unity 缓存时删除。连接、重新配置和诊断命令见 [MCP 接入说明](UNITY_MCP_SETUP.md)。
