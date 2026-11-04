[CmdletBinding()]
param(
    [string]$Url = 'http://127.0.0.1:8090/mcp'
)

$ErrorActionPreference = 'Stop'
Get-Command codex -ErrorAction Stop | Out-Null
Get-Command uv -ErrorAction Stop | Out-Null

# Codex owns TOML updates, preserving the other configured MCP servers.
& codex mcp add unityMCP --url $Url
if ($LASTEXITCODE -ne 0) { throw 'Could not register unityMCP in Codex.' }

$checker = Join-Path $PSScriptRoot 'unity_mcp.py'
& uv run $checker --url $Url check
if ($LASTEXITCODE -ne 0) {
    throw 'Codex is configured, but Unity validation failed. Start the Unity MCP HTTP server and session, then rerun.'
}
