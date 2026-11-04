# Codex 与 Unity MCP 连接

2026-09-20 已在本机完成真实协议与游戏验证。

## 两条独立链路

- 开发控制：Codex → HTTP MCP `http://127.0.0.1:8090/mcp` → MCP for Unity → Unity Editor。
- 游戏控制：Python 运动想象分类 → UDP `127.0.0.1:7777` → ExternalMazeInput → MazeGameController。

MCP 可读取场景、控制台、编辑器状态，启动/停止 Play，截图和执行编辑器工具。它不替代 EEG 推理或 UDP 移动接口。

## 本机已验证配置

| 项目 | 值 |
| --- | --- |
| Unity 工程 | `D:/xht/BCI-Maze game/Maze-game` |
| Unity 版本 | `2022.3.57f1c1` |
| 插件和 Python 服务包版本 | `10.1.0` |
| Unity 实例 | `Maze-game@5a0c75a1d2e2a765` |
| Codex 服务名 | `unityMCP` |
| HTTP 地址 | `http://127.0.0.1:8090/mcp` |
| 已发现工具 / 资源 | 48 / 19 |

`Packages/manifest.json` 已有 CoplayDev 插件，`packages-lock.json` 当前锁定提交 `c14de1e6dc01ab42d2bb358730cff954bce0ce6b`。不要为了接入而重复安装插件。manifest 跟踪 main，主动升级前应评估兼容性。

本机 `%USERPROFILE%/.codex/config.toml` 已有：

```toml
[mcp_servers.unityMCP]
url = "http://127.0.0.1:8090/mcp"
```

## 启动与配置

1. 在 Unity Hub 中打开上述工程，等待编译完成。
2. 打开 Unity 的 MCP for Unity 窗口，将本地 HTTP 地址设为 `http://127.0.0.1:8090`，启动服务及 Session。看到 Session connected 后再验证。
3. 在仓库根目录执行以下命令。脚本用 Codex 官方 CLI 写入用户配置，再用 MCP SDK 实际查询 Unity；需要已安装 `codex` 和 `uv`。

```powershell
.\scripts\setup_unity_mcp.ps1
```

只验证、不修改配置：

```powershell
uv run scripts/unity_mcp.py --output outputs/unity_mcp/check.json check
```

`uv` 按脚本内声明安装隔离的 `mcp==1.30.0`，不改变 EEG 的 `.venv` / `.venv-mi`。首次运行可能需要联网下载依赖。

当前任务创建时没有加载原生 Unity MCP 工具列表，但通过以上 MCP SDK 入口已经成功控制 Unity。若 Codex 工具列表仍没有 unityMCP，重启 Codex 后重新进入项目，再验证工具是否加载。不要将“配置存在”当作“工具已加载”。在当前任务中可以继续通过该脚本操作真实 MCP 服务。

## 常用操作

以下 JSON 引号形式已在本机 PowerShell 7 验证。

```powershell
uv run scripts/unity_mcp.py tools
uv run scripts/unity_mcp.py read mcpforunity://instances
uv run scripts/unity_mcp.py read mcpforunity://editor/state
uv run scripts/unity_mcp.py call manage_scene '{"action":"get_active"}'
uv run scripts/unity_mcp.py call read_console '{"action":"get","types":["error","warning"],"count":"20"}'
uv run scripts/unity_mcp.py call manage_editor '{"action":"play"}'
uv run scripts/unity_mcp.py call manage_editor '{"action":"stop"}'
```

多工程同时连接时，在子命令前使用 `--instance 'Name@hash'` 明确选择目标；实例值以实时 instances 资源为准。执行修改操作前先读编辑器状态和对应工具 schema。`check` 为只读检查，不会自动进入 Play 或清空 Console。

## 排障

- HTTP 无法连接：确认 Unity 中服务启动，地址为 8090，检查 `Maze-game/Library/MCPForUnity/Logs/server-launch-8090.log`。
- HTTP 健康但无 Unity 实例：启动插件 Session，检查编译或域重载是否尚未结束。
- 对 `/mcp` 普通 GET 返回 406：不能据此判断服务失效，MCP 需要正确的初始化及 Accept 头；使用本项目 SDK 检查。
- 该本地服务未启用 OAuth，不需要执行 `codex mcp login unityMCP`；OAuth 发现 URL 的 404 不能替代实际 MCP 握手结果。
- MCP 可读编辑器但 UDP 超时：UDP 接收组件在 Play 时启动，确认已进入 Play 且端口为 7777。
- 工具或资源返回失败时诊断脚本以非零状态退出；报告中的 Console 警告仍需人工判断。

配置方式参考 [OpenAI 官方 MCP 文档](https://developers.openai.com/codex/mcp)。游戏验收与限制见 [Unity 实现检查报告](UNITY_IMPLEMENTATION_REVIEW_20260920.md)。
