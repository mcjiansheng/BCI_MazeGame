# Unity BCI Maze 接口说明

## 项目与场景

- Unity 版本：2022.3.57f1c1
- 场景：Scenes/SampleScene.unity
- 迷宫控制脚本：Scripts/MazeGameController.cs
- 外部输入脚本：Scripts/ExternalMazeInput.cs
- 默认迷宫：61 × 61
- 默认监听：127.0.0.1:7777

打开 SampleScene.unity 后点击 Play 即可运行。场景中应包含 MazeGameController 和 ExternalMazeInput；如果复制到新场景，需确认两个组件都已挂载并启用。

## 游戏操作

| 输入 | 行为 |
| --- | --- |
| W / ↑ | 向上 |
| S / ↓ | 向下 |
| A / ← | 向左 |
| D / → | 向右 |
| R | 重新生成迷宫 |
| H | 计算并显示 BFS 最短路长度 |
| V | 在跟随视角和全图视角之间切换 |

迷宫由迭代 DFS 生成，起点为 (1, 1)，终点为 (width - 2, height - 2)。墙体不可通行；到达终点后会显示完成状态。跟随相机和全图相机都由 MazeGameController 管理。

## UDP 协议

Unity 默认只绑定回环地址，因此本机程序可以发送，局域网其他主机不能直接访问。支持纯文本和 JSON 两种载荷，方向会被归一化为以下命令：

| 规范命令 | 可接受别名 |
| --- | --- |
| left | left、west、a、move_left |
| right | right、east、d、move_right |
| up | up、north、w、move_up、forward |
| down | down、south、s、move_down、backward |

纯文本示例：

~~~powershell
$client = New-Object System.Net.Sockets.UdpClient
$bytes = [Text.Encoding]::UTF8.GetBytes("left")
$client.Send($bytes, $bytes.Length, "127.0.0.1", 7777)
$client.Close()
~~~

JSON 示例：

~~~json
{"command":"left"}
~~~

Unity 会返回 JSON ACK：

~~~json
{"command":"left","accepted":true}
~~~

accepted 为 true 表示该命令已被游戏逻辑接受并执行；false 通常表示命令非法、迷宫墙体阻挡、已经完成或当前状态不允许移动。ACK 不代表模型预测正确，只代表 Unity 对这次移动的处理结果。

## 节流与线程语义

键盘输入使用 MazeGameController 的 moveCooldownSeconds，默认值为 0.12 秒。外部 UDP 路径由接收器直接提交到游戏逻辑，不应依赖这个键盘冷却值；发送方应自行控制发送频率，建议相邻命令至少间隔 120–150 ms，避免连续预测造成过快移动。

代码中有两种外部调用方式，返回值含义不同：

- MazeGameController.SubmitExternalCommand：同步调用，返回命令是否被迷宫逻辑接受。
- ExternalMazeInput.SubmitCommand：当组件实例存在时把命令加入主线程队列，返回 true 只表示成功入队，不等于角色已经移动；Update 中实际处理后才知道是否接受。
- maxCommandsPerFrame 默认是 4，只限制 ExternalMazeInput 的主线程队列处理速度。

因此，外部程序若需要可靠记录，应以收到的 UDP ACK 为准，并同时记录发送时间、预测类别、置信度和迷宫状态。

## 与 Python 解码脚本联调

先在 Unity 中打开 SampleScene.unity 并点击 Play，再运行仓库根目录的联调脚本：

~~~powershell
python scripts/run_bci_maze_integration.py --host 127.0.0.1 --port 7777
~~~

脚本默认读取：

- BCIC2a A01 的 E 测试数据；
- outputs/checkpoints/bcic2a/se_mhaf_conformer_final/A01.pt；
- 模型 se_mhaf_conformer_final_logvar；
- 标签 0、1、2 分别映射为 left、right、up，标签 3 不发送移动。

脚本发送方向指令、等待 ACK，并把结果写入 outputs/bci_maze_integration_test.json。该过程是端到端冒烟测试，不是通关率或在线准确率评估；模型预测到墙体方向时，Unity 正确返回 accepted=false 是预期行为。

## 调试建议

1. Unity Console 先确认 ExternalMazeInput 已启动并监听 7777。
2. 用纯文本 left 或 JSON {"command":"left"} 做最小 UDP 测试。
3. 检查 ACK 的 accepted 字段，不要只检查发送函数是否返回。
4. 再运行 Python 联调脚本，确认 checkpoint、预处理目录和 Unity 端口一致。
5. 如需改变端口或监听地址，同时修改场景组件参数、Python 脚本参数和本文档。

## 安全与研究边界

默认回环监听是为了降低误接收风险；脚本没有身份认证、加密或重放保护，不适合直接暴露到公网。若要跨主机使用，应增加鉴权、访问控制和异常速率限制。

Unity 的 accepted 只说明移动是否被迷宫规则接受，不能作为脑机解码真值。研究记录应区分“模型预测”“命令发送”“Unity 接收”“移动接受”“撞墙/无效”和“到达终点”等事件。
