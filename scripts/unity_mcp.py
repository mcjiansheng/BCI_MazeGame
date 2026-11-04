# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.30.0"]
# ///
"""Inspect or call the existing Unity MCP server using the official MCP SDK."""

import argparse
import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def unpack(result):
    value = result.model_dump(mode="json", exclude_none=True)
    if value.get("isError"):
        raise RuntimeError(str(value))
    for item in value.get("contents", value.get("content", [])):
        if "text" in item:
            try:
                item["decoded"] = json.loads(item["text"])
                del item["text"]
            except json.JSONDecodeError:
                pass
            decoded = item.get("decoded")
            if isinstance(decoded, dict) and decoded.get("success") is False:
                raise RuntimeError(str(decoded))
    return value


async def run(args):
    async with streamable_http_client(args.url) as (read, write, _):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=args.timeout)
        ) as session:
            initialized = await session.initialize()
            if args.instance:
                selected = await session.call_tool(
                    "set_active_instance", {"instance": args.instance}
                )
                if selected.isError:
                    raise RuntimeError(str(unpack(selected)))
            if args.command == "tools":
                return unpack(await session.list_tools())
            if args.command == "read":
                return unpack(await session.read_resource(args.uri))
            if args.command == "call":
                parameters = json.loads(args.arguments)
                if not isinstance(parameters, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                result = await session.call_tool(args.name, parameters)
                if result.isError:
                    raise RuntimeError(str(unpack(result)))
                return unpack(result)
            resources = await session.list_resources()
            tools = await session.list_tools()
            report = {
                "url": args.url,
                "server": initialized.serverInfo.model_dump(),
                "tool_count": len(tools.tools),
                "resource_count": len(resources.resources),
                "resources": {},
            }
            available = {item.name: item.uri for item in resources.resources}
            for name in ("custom_tools", "unity_instances", "project_info", "editor_state"):
                if name not in available:
                    raise RuntimeError(f"Required Unity resource missing: {name}")
                report["resources"][name] = unpack(
                    await session.read_resource(available[name])
                )
            instances = report["resources"]["unity_instances"]["contents"][0]["decoded"]
            if not instances.get("instance_count"):
                raise RuntimeError("MCP server is running but no Unity Editor is connected")
            for name, parameters in (
                ("manage_scene", {"action": "get_active"}),
                ("read_console", {"action": "get", "types": ["error", "warning"],
                                  "count": "20", "include_stacktrace": True}),
            ):
                result = await session.call_tool(name, parameters)
                if result.isError:
                    raise RuntimeError(str(unpack(result)))
                report[name] = unpack(result)
            return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090/mcp")
    parser.add_argument("--instance", help="Exact Name@hash from the instances resource")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Read server, project, editor, scene and console")
    commands.add_parser("tools", help="List tool schemas before making a call")
    read = commands.add_parser("read")
    read.add_argument("uri")
    call = commands.add_parser("call")
    call.add_argument("name")
    call.add_argument("arguments", nargs="?", default="{}", help="JSON object")
    args = parser.parse_args()
    try:
        result = asyncio.run(asyncio.wait_for(run(args), timeout=args.timeout * 8))
    except Exception as error:
        def details(exception):
            children = getattr(exception, "exceptions", ())
            return "; ".join(details(child) for child in children) if children else str(exception)
        print(f"Unity MCP failed: {details(error)}", file=sys.stderr)
        return 1
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
