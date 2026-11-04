# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.30.0"]
# ///
"""Verify a saved SE-MHAF model, real-time EEG replay and the actual Unity game.

Requires the project's local Unity MCP server. Results explicitly identify the
replayed recording; hardware transport testing remains a separate operation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import time
import uuid

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from unity_mcp import unpack

ROOT = Path(__file__).resolve().parents[1]


def udp(command, expected=None):
    stamp = uuid.uuid4().hex
    payload = {"command": command, "ts": stamp}
    if expected is not None:
        payload["expected"] = expected
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.settimeout(2)
        connection.connect(("127.0.0.1", 7777))
        connection.send(json.dumps(payload).encode())
        reply = json.loads(connection.recv(8192))
    assert reply["echo_ts"] == stamp and reply["command"] == command, reply
    return reply


def expected_state(nav):
    return {key: nav[key] for key in (
        "maze_generation", "x", "y", "heading", "steps", "paused", "state", "pending_turn",
    )}


async def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"input": "recorded_EEG_replay", "model": str(args.model.resolve()),
              "recording": str(args.replay.resolve()), "duration_seconds": args.duration}
    async with streamable_http_client("http://127.0.0.1:8090/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def code(source):
                result = unpack(await session.call_tool("execute_code", {"action": "execute", "code": source}))
                return result["structuredContent"]["data"]["result"]

            await session.call_tool("manage_editor", {"action": "play"})
            await asyncio.sleep(1)
            previous = await code("var g = MazeGameController.Instance; return new { speed=g.moveSpeed, turn=g.turnSeconds, wait=g.junctionWaitSeconds, seed=g.fixedSeed, fixedSeed=g.useFixedSeed };")
            try:
                report["unity_condition_checks"] = await code((ROOT / "tests/unity/ConditionalControlSmoke.cs").read_text(encoding="utf-8"))
                state = (await asyncio.to_thread(udp, "status"))["navigation"]
                stale = expected_state(state)
                stale["maze_generation"] -= 1
                rejected = await asyncio.to_thread(udp, "left", stale)
                assert rejected["accepted"] is False, rejected
                assert rejected["navigation"]["pending_turn"] == state["pending_turn"], rejected
                report["stale_udp_request_rejected"] = rejected
                await code("var g=MazeGameController.Instance; g.useFixedSeed=true; g.fixedSeed=42; g.moveSpeed=8; g.turnSeconds=0.1f; g.junctionWaitSeconds=4; g.GenerateMaze(); return g.GetStatus();")
                log = output / "decisions.jsonl"
                if log.exists():
                    raise FileExistsError(f"Choose a fresh output directory: {output}")
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(ROOT / "src")
                environment["PYTHONUTF8"] = "1"
                interpreter = ROOT / ".venv/Scripts/python.exe"
                with (output / "online.log").open("w", encoding="utf-8") as stdout:
                    process = await asyncio.create_subprocess_exec(
                        str(interpreter), "-u", str(ROOT / "scripts/online_lk_mi.py"),
                        "--model", str(args.model.resolve()), "--backend", "replay", "--replay", str(args.replay.resolve()),
                        "--device", "cuda", "--duration", str(args.duration), "--unity-host", "127.0.0.1",
                        "--allow-unvalidated-control", "--log", str(log), cwd=ROOT, env=environment,
                        stdout=stdout, stderr=asyncio.subprocess.STDOUT,
                    )
                    try:
                        await asyncio.wait_for(process.wait(), args.duration + 45)
                    except BaseException:
                        if process.returncode is None:
                            process.terminate()
                            await process.wait()
                        raise
                assert process.returncode == 0, f"online exited {process.returncode}: {output / 'online.log'}"
                records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
                motions = [row for row in records if row.get("event") == "unity" and row.get("confirmed") and row.get("command") in {"left", "right", "forward"}]
                queued = [row for row in records if row.get("unity", {}).get("queued")]
                tokens = [tuple(row["unity"]["visit_token"]) for row in queued]
                assert len(tokens) == len(set(tokens)), "A visit received multiple intents"
                assert motions, "No model decision reached Unity; inspect online.log"
                final = (await asyncio.to_thread(udp, "status"))["navigation"]
                assert final["steps"] > 0, final
                report.update({"passed": True, "confirmed_motion_count": len(motions), "confirmed_motions": motions,
                               "queued_decisions": len(queued), "unique_visit_tokens": len(set(tokens)),
                               "segment_changes": sum(row.get("reason") == "segment_changed" for row in records),
                               "final_navigation": final})
            finally:
                await code("var g=MazeGameController.Instance; "
                           f"g.moveSpeed={previous['speed']}f; g.turnSeconds={previous['turn']}f; g.junctionWaitSeconds={previous['wait']}f; "
                           f"g.fixedSeed={previous['seed']}; g.useFixedSeed={str(previous['fixedSeed']).lower()}; "
                           'g.GenerateMaze(); g.TryMove("pause"); return g.GetStatus();')
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"confirmed_motions", "stale_udp_request_rejected"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60)
    arguments = parser.parse_args()
    if arguments.duration < 10:
        parser.error("duration must be at least 10 seconds")
    asyncio.run(run(arguments))
