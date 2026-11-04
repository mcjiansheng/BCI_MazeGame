# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.30.0"]
# ///
"""Run navigation regressions and a real UDP relative-turn playthrough in Unity Play mode."""
import asyncio
import json
import socket
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from unity_mcp import unpack

ROOT = Path(__file__).resolve().parents[1]


async def main():
    report = {}
    async with streamable_http_client("http://127.0.0.1:8090/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def code(source):
                result = unpack(await session.call_tool("execute_code", {"action": "execute", "code": source}))
                return result["structuredContent"]["data"]["result"]

            report["navigation"] = await code((ROOT / "tests/unity/NavigationSmoke.cs").read_text(encoding="utf-8"))
            previous = await code("var g = MazeGameController.Instance; return new { speed = g.moveSpeed, turn = g.turnSeconds, wait = g.junctionWaitSeconds, seed = g.fixedSeed, fixedSeed = g.useFixedSeed };")
            try:
                route = await code("var g = MazeGameController.Instance; g.useFixedSeed = true; g.fixedSeed = 42; g.moveSpeed = 28; g.turnSeconds = 0.05f; g.junctionWaitSeconds = 0.15f; g.GenerateMaze(); g.TryMove(\"pause\"); return g.GetHintPath().Select(p => new {x=p.x,y=p.y}).ToArray();")
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.settimeout(5)

                    def send(command):
                        stamp = time.time()
                        sock.sendto(json.dumps({"command": command, "ts": stamp}).encode(), ("127.0.0.1", 7777))
                        ack = json.loads(sock.recvfrom(8192)[0])
                        assert ack["accepted"], ack
                        assert abs(ack["echo_ts"] - stamp) < 1e-5, ack
                        return ack["navigation"]

                    await asyncio.to_thread(send, "heartbeat")
                    assert (await asyncio.to_thread(send, "left"))["pending_turn"] == "left"
                    assert (await asyncio.to_thread(send, "right"))["pending_turn"] == "right"
                    assert (await asyncio.to_thread(send, "down"))["pending_turn"] == "uturn"
                    assert (await asyncio.to_thread(send, "cancel"))["pending_turn"] == ""
                    next_cell = {(a["x"], a["y"]): (b["x"], b["y"]) for a, b in zip(route, route[1:])}
                    headings = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
                    await asyncio.to_thread(send, "resume")
                    commands = []
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        state = await asyncio.to_thread(send, "status")
                        if state["state"] == "Complete":
                            break
                        if state["state"] == "Junction":
                            x, y = state["x"], state["y"]
                            nx, ny = next_cell[(x, y)]
                            relative = (headings[(nx - x, ny - y)] - state["heading"]) % 4
                            if relative in (1, 3):
                                command = "right" if relative == 1 else "left"
                                await asyncio.to_thread(send, command)
                                commands.append(command)
                            else:
                                assert relative == 0, state
                        await asyncio.sleep(0.02)
                    assert state["state"] == "Complete", state
                    assert state["steps"] == len(route) - 1, state
                    report["udp"] = {"passed": True, "state": state, "decision_commands": commands,
                                     "heartbeat": True, "timestamp_echo": True, "buffer_replace_cancel": True,
                                     "legacy_down_alias": True}
                report["screenshot"] = unpack(await session.call_tool("manage_camera", {
                    "action": "screenshot", "screenshot_file_name": "maze-3d-complete.png",
                    "output_folder": "Library/MCPForUnity/Verification"}))
                await asyncio.sleep(1)
            finally:
                await code(
                    "var g = MazeGameController.Instance; "
                    f"g.moveSpeed = {previous['speed']}f; g.turnSeconds = {previous['turn']}f; "
                    f"g.junctionWaitSeconds = {previous['wait']}f; g.fixedSeed = {previous['seed']}; "
                    f"g.useFixedSeed = {str(previous['fixedSeed']).lower()}; "
                    'g.GenerateMaze(); g.TryMove("pause"); return g.GetStatus();'
                )
    output = ROOT / "outputs/unity_mcp/first-person-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"navigation_checks": report["navigation"]["passed"], "udp": report["udp"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
