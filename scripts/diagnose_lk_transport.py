"""Capture the LK-Mini TCP transport independently of BrainFlow decoding."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import socket
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bci_maze.lk_mini.source import build_all_channel_commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--recv-size", type=int, default=65536)
    parser.add_argument("--recv-buffer", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, choices=(250, 500, 1000), default=250,
                        help="expected rate per channel; --configure currently sets 250 Hz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.recv_size <= 0 or args.recv_buffer < 0:
        raise ValueError("duration and receive size must be positive; buffer must be non-negative")
    if args.configure and args.sample_rate != 250:
        raise ValueError("--configure requires --sample-rate 250")
    args.output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = {"ip": args.ip, "seconds": args.seconds, "configure": args.configure,
              "recv_size": args.recv_size, "http": [], "bytes_per_second": [0] * args.seconds}
    report["requested_recv_buffer"] = args.recv_buffer
    raw = bytearray()
    phase = "route"

    def request(path, data=None):
        nonlocal phase
        phase = path
        payload = None if data is None else json.dumps(data).encode()
        query = urllib.request.Request(f"http://{args.ip}{path}", data=payload)
        if data is not None:
            query.add_header("Content-Type", "application/json")
        started = time.monotonic()
        entry = {"path": path}
        try:
            with opener.open(query, timeout=5) as response:
                body = response.read().decode(errors="replace")
                entry.update(status=response.status, body=body)
        except Exception as exc:
            entry["error"] = repr(exc)
            raise
        finally:
            entry["seconds"] = time.monotonic() - started
            report["http"].append(entry)
        return body

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((args.ip, 80))
        local_ip = probe.getsockname()[0]
    report["local_ip"] = local_ip
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            if args.recv_buffer:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.recv_buffer)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((local_ip, args.port))
            server.listen(1)
            server.settimeout(5)
            request("/board")
            request("/tcp", {"ip": local_ip, "port": args.port, "output": "raw",
                             "delimiter": True, "latency": 10000})
            connection, peer = server.accept()
            with connection:
                if args.recv_buffer:
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.recv_buffer)
                report["recv_buffer"] = connection.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                report["peer"] = list(peer)
                connection.settimeout(0.25)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if args.configure:
                    for command in ("~4", "d", "~6", build_all_channel_commands(range(16), 24)):
                        request("/command", {"command": command})
                        time.sleep(0.5)
                request("/stream/start")
                phase = "receive"
                started = time.monotonic()
                next_update = 0
                last_data = started
                report["max_idle_seconds"] = 0.0
                while time.monotonic() - started < args.seconds:
                    try:
                        block = connection.recv(args.recv_size)
                    except socket.timeout:
                        block = None
                    elapsed = time.monotonic() - started
                    report["max_idle_seconds"] = max(
                        report["max_idle_seconds"], time.monotonic() - last_data
                    )
                    if block == b"":
                        report["peer_closed_at_seconds"] = elapsed
                        break
                    if block:
                        last_data = time.monotonic()
                        raw.extend(block)
                        report["bytes_per_second"][min(int(elapsed), args.seconds - 1)] += len(block)
                    if elapsed >= next_update:
                        print(f"{elapsed:.1f}s received={len(raw)} bytes", flush=True)
                        next_update += 3
                report["receive_seconds"] = time.monotonic() - started
    except Exception as exc:
        report["error"] = repr(exc)
        report["failed_phase"] = phase
    finally:
        try:
            request("/stream/stop")
        except Exception as exc:
            report["stop_error"] = repr(exc)

    # Diagnostic frame counting only; classification continues to use BrainFlow.
    offset, frames, discarded = 0, [], 0
    while offset + 33 <= len(raw):
        if raw[offset] == 0xA0 and 0xC0 <= raw[offset + 32] <= 0xC6:
            frames.append(raw[offset + 1])
            offset += 33
        else:
            offset += 1
            discarded += 1
    report.update({"bytes": len(raw), "valid_33_byte_frames": len(frames),
                   "discarded_bytes": discarded, "trailing_bytes": len(raw) - offset,
                   "frame_counter_steps": dict(Counter((b - a) % 256 for a, b in zip(frames, frames[1:]))),
                   "first_frame_counters": frames[:32]})
    steps = [(b - a) % 256 for a, b in zip(frames, frames[1:])]
    phase_offset = steps[0] if steps and steps[0] in (0, 1) else 0
    sequence_ok = bool(steps) and all(step == (i + phase_offset) % 2 for i, step in enumerate(steps))
    elapsed = report.get("receive_seconds", args.seconds)
    pair_rate = len(frames) / 2 / max(elapsed, 1e-9)
    report.update(expected_sample_rate=args.sample_rate, estimated_pair_rate=pair_rate,
                  frame_sequence_ok=sequence_ok)
    # A short burst followed by silence must not pass a sustained-stream check.
    report["sustained_stream_ok"] = bool(
        "error" not in report and "stop_error" not in report
        and "peer_closed_at_seconds" not in report and not discarded
        and sequence_ok and abs(pair_rate - args.sample_rate) / args.sample_rate <= 0.15
        and report.get("max_idle_seconds", float("inf")) <= 1.0
    )
    (args.output / "transport.bin").write_bytes(raw)
    (args.output / "transport.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["sustained_stream_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
