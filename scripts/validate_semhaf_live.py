"""Run the real online classifier with timed HTTP stream-stop recovery checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_new_records(path: Path, offset: int = 0) -> tuple[list[dict], int]:
    records = []
    if not path.exists():
        return records, offset
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line or not line.endswith("\n"):
                return records, start
            records.append(json.loads(line))


def build_online_command(args, decision_path: Path, transport_path: Path) -> list[str]:
    return [str(args.python), "-X", "utf8", "-u", str(ROOT / "scripts/online_lk_mi.py"),
            "--model", str(args.model.resolve()), "--backend", "brainflow", "--device", args.device,
            "--duration", str(args.duration), "--ip-address", args.ip_address,
            "--ip-port", str(args.ip_port), "--no-data-timeout", str(args.no_data_timeout),
            "--log", str(decision_path), "--transport-log", str(transport_path)]


def inject_stop(ip_address: str, *, scheduled: float, first_data: float, previous_chunk: dict) -> dict:
    now = time.monotonic()
    record = {"event": "device_stream_stop", "scheduled_after_first_data": scheduled,
              "request_started_monotonic": now, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_after_first_data": now - first_data,
              "segment_before": previous_chunk["segment_id"],
              "last_chunk_before_request": previous_chunk}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{ip_address}/stream/stop", timeout=2.0) as response:
            record["http_status"] = response.status
            record["response_body"] = response.read(4096).decode("utf-8", errors="replace")
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["response_finished_monotonic"] = time.monotonic()
    return record


def terminate_owned_process(process) -> dict:
    """Abnormal controller exit must also stop its owned BrainFlow child on Windows."""
    record = {"pid": process.pid, "monotonic": time.monotonic(), "action": "already_exited"}
    if process.poll() is not None:
        return record
    try:
        if os.name == "nt":
            command = ["taskkill", "/PID", str(process.pid), "/T", "/F"]
            record.update(action="terminate_owned_process_tree", command=command)
            result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=15)
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        else:
            record["action"] = "terminate_owned_process"
            process.terminate()
        process.wait(timeout=10)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
            record["parent_kill_fallback"] = True
    return record


def summarize_live(transport: list[dict], decisions: list[dict], injections: list[dict],
                   *, requested_duration: float, returncode: int, wear_state: str,
                   missed_injections: list[float] | None = None, expected_injection_count: int | None = None) -> dict:
    chunks = [record for record in transport if record.get("event") == "chunk"]
    starts = [record for record in transport if record.get("event") == "start"]
    summaries = [record for record in transport if record.get("event") == "summary"]
    summary = summaries[-1] if summaries else {}
    started = starts[0] if starts else {}
    sample_rate = started.get("sample_rate", 250)
    expected_samples = started.get("window_samples", 1000)
    windows = [record for record in decisions if "window_samples" in record]
    source_events = {}
    for record in transport:
        for event in record.get("recovery_events", []):
            source_events[(event.get("type"), event.get("time_monotonic"), event.get("segment_id"))] = event
    segments = []
    for segment in dict.fromkeys(record["segment_id"] for record in chunks):
        data = [record for record in chunks if record["segment_id"] == segment]
        segment_windows = [record for record in windows if record["segment_id"] == segment]
        first_window = segment_windows[0] if segment_windows else None
        valid = bool(first_window and first_window["window_samples"] == expected_samples
                     and first_window["segment_samples_received"] >= expected_samples
                     and first_window["window_start"] >= data[0]["timestamp_start"] - 1e-6
                     and abs(first_window["window_span_seconds"] - expected_samples / sample_rate) < 2.5 / sample_rate)
        segments.append({"segment_id": segment, "sample_count": sum(record["sample_count"] for record in data),
                         "first_read_monotonic": data[0]["monotonic"], "last_read_monotonic": data[-1]["monotonic"],
                         "first_sample_timestamp": data[0]["timestamp_start"], "last_sample_timestamp": data[-1]["timestamp_end"],
                         "continuity_errors": sum(record.get("continuity_errors", 0) for record in data),
                         "first_complete_window": first_window, "fresh_window_verified": valid,
                         "quality_rejections": sum(record.get("reason") == "bad_signal" for record in segment_windows),
                         "inference_calls": sum(record.get("inference_invoked") is True for record in segment_windows)})
    injection_results = []
    for index, injection in enumerate(injections):
        began = injection["request_started_monotonic"]
        until = injections[index + 1]["request_started_monotonic"] if index + 1 < len(injections) else float("inf")
        new_data = next((record for record in chunks if began <= record["monotonic"] < until
                         and record["segment_id"] != injection["segment_before"]), None)
        recovery_end = new_data["monotonic"] if new_data else until
        accepted_during_gap = sum(record.get("accepted") is True and began <= record.get("monotonic", -1) < recovery_end for record in windows)
        entry = next((record for record in segments if new_data and record["segment_id"] == new_data["segment_id"]), None)
        first_window = entry["first_complete_window"] if entry else None
        matching_events = [event for event in source_events.values()
                           if new_data and event.get("segment_id") == new_data["segment_id"]
                           and began <= event.get("time_monotonic", -1) < until]
        event_types = {event.get("type") for event in matching_events}
        injection_results.append({
            **injection, "new_segment_id": None if new_data is None else new_data["segment_id"],
            "http_injection_succeeded": not injection.get("error") and 200 <= injection.get("http_status", 0) < 300,
            "recovery_events_matched": {"disconnected", "recovered"}.issubset(event_types),
            "matching_recovery_events": matching_events,
            "first_new_segment_chunk": new_data,
            "new_segment_samples": 0 if entry is None else entry["sample_count"],
            "first_new_sample_delay_seconds": None if new_data is None else new_data["monotonic"] - began,
            "first_complete_window": first_window,
            "first_complete_window_delay_seconds": None if first_window is None else first_window["monotonic"] - began,
            "fresh_window_verified": bool(entry and entry["fresh_window_verified"]),
            "accepted_between_injection_and_new_segment": accepted_during_gap,
            "quality_rejections_after_injection": sum(record.get("reason") == "bad_signal" and began <= record.get("monotonic", -1) < until for record in windows),
        })
    recovery_intervals = []
    recovery_start = None
    for record in transport:
        if record.get("event") != "state":
            continue
        if record.get("recovering") and recovery_start is None:
            recovery_start = record["monotonic"]
        elif not record.get("recovering") and recovery_start is not None:
            recovery_intervals.append((recovery_start, record["monotonic"]))
            recovery_start = None
    if recovery_start is not None:
        recovery_intervals.append((recovery_start, summary.get("capture_end_monotonic", float("inf"))))
    accepted_during_recovery = sum(record.get("accepted") is True and any(start <= record.get("monotonic", -1) < stop for start, stop in recovery_intervals) for record in windows)
    tail = summary.get("tail_silence_seconds")
    expected_injection_count = len(injections) + len(missed_injections or []) if expected_injection_count is None else expected_injection_count
    injection_count_matches = len(injections) == expected_injection_count and not missed_injections
    first_data_delay = chunks[0]["monotonic"] - started.get("monotonic", chunks[0]["monotonic"]) if chunks else None
    completed = returncode == 0 and not summary.get("error") and summary.get("acquisition_elapsed", 0) >= requested_duration - 0.5
    passed = bool(completed and chunks and injection_count_matches and tail is not None and tail < 1.0
                  and first_data_delay is not None and first_data_delay <= 15.0
                  and segments and segments[0]["fresh_window_verified"]
                  and not sum(record["continuity_errors"] for record in segments)
                  and not sum(record.get("nonfinite_data_values", 0) + record.get("nonfinite_timestamps", 0) for record in chunks)
                  and not accepted_during_recovery
                  and all(record["http_injection_succeeded"] and record["recovery_events_matched"]
                          and record["fresh_window_verified"] and record["accepted_between_injection_and_new_segment"] == 0 for record in injection_results))
    return {"purpose": "unworn transport, recovery and production quality-gate validation",
            "wear_state": wear_state, "MI_accuracy": None, "labeled_human_trials": 0,
            "requested_duration": requested_duration, "online_returncode": returncode,
            "capture_completed": completed, "transport_and_rewarm_passed": passed,
            "samples": sum(record["sample_count"] for record in chunks), "segments": segments,
            "first_data_monotonic": None if not chunks else chunks[0]["monotonic"],
            "last_data_monotonic": None if not chunks else chunks[-1]["monotonic"],
            "tail_silence_seconds": tail, "data_at_end": bool(tail is not None and tail < 1.0),
            "first_data_delay_seconds": first_data_delay,
            "planned_injection_count": expected_injection_count, "injection_count_matches": bool(injection_count_matches),
            "decision_windows": len(windows), "quality_rejections": sum(record.get("reason") == "bad_signal" for record in windows),
            "quality_gate_count": len(windows),
            "inference_invoked_count": sum(record.get("inference_invoked") is True for record in windows),
            "inference_calls": sum(record.get("inference_invoked") is True for record in windows),
            "nonfinite_data_values": sum(record.get("nonfinite_data_values", 0) for record in chunks),
            "nonfinite_timestamps": sum(record.get("nonfinite_timestamps", 0) for record in chunks),
            "accepted_decisions": sum(record.get("accepted") is True for record in windows),
            "accepted_during_recovery": accepted_during_recovery,
            "recovery_intervals_monotonic": recovery_intervals, "injections": injection_results,
            "missed_injections": missed_injections or [], "transport_summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--stop-stream-at", default="25,75", help="seconds after the first non-empty online read")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--ip-address", default="192.168.4.1")
    parser.add_argument("--ip-port", type=int, default=12345)
    parser.add_argument("--no-data-timeout", type=float, default=3.0)
    parser.add_argument("--wear-state", choices=("unworn", "worn", "unspecified"), default="unworn")
    args = parser.parse_args(argv)
    scheduled = sorted(float(value) for value in args.stop_stream_at.split(",") if value.strip())
    if args.duration <= 0 or any(not 0 < value < args.duration for value in scheduled) or len(set(scheduled)) != len(scheduled):
        parser.error("Use a positive duration and unique injection times strictly inside it")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path, transport_path = args.output_dir / "decisions.jsonl", args.output_dir / "transport.jsonl"
    if any(path.exists() for path in (decisions_path, transport_path)):
        parser.error("Use a fresh output directory to keep this acquisition separate")
    command = build_online_command(args, decisions_path, transport_path)
    (args.output_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    records, injections, offset = [], [], 0
    pending = list(scheduled)
    first_data = None
    last_chunk = None
    controller_error = ""
    controller_cleanup = []
    controller_started = time.monotonic()
    with (args.output_dir / "stdout.log").open("wb") as stdout, (args.output_dir / "stderr.log").open("wb") as stderr, (args.output_dir / "injections.jsonl").open("w", encoding="utf-8") as injection_log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        try:
            while process.poll() is None:
                new, offset = read_new_records(transport_path, offset)
                records.extend(new)
                for record in new:
                    if record.get("event") == "chunk":
                        first_data = record["monotonic"] if first_data is None else first_data
                        last_chunk = record
                if first_data is not None and pending and time.monotonic() - first_data >= pending[0]:
                    event = inject_stop(args.ip_address, scheduled=pending.pop(0), first_data=first_data, previous_chunk=last_chunk)
                    injections.append(event)
                    injection_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    injection_log.flush()
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                if time.monotonic() - controller_started > args.duration + 90:
                    raise TimeoutError("Online subprocess exceeded duration plus 90-second cleanup allowance")
                time.sleep(0.1)
        except (KeyboardInterrupt, Exception) as error:
            controller_error = f"{type(error).__name__}: {error}"
            controller_cleanup.append(terminate_owned_process(process))
        returncode = process.wait()
    new, _ = read_new_records(transport_path, offset)
    records.extend(new)
    decisions, _ = read_new_records(decisions_path)
    report = summarize_live(records, decisions, injections, requested_duration=args.duration,
                            returncode=returncode, wear_state=args.wear_state, missed_injections=pending,
                            expected_injection_count=len(scheduled))
    report["controller_error"] = controller_error
    report["controller_cleanup"] = controller_cleanup
    report["model"] = str(args.model.resolve())
    if controller_error:
        report["transport_and_rewarm_passed"] = False
    destination = args.output_dir / "live_report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("capture_completed", "transport_and_rewarm_passed", "samples", "decision_windows", "quality_rejections", "inference_calls", "accepted_during_recovery", "tail_silence_seconds")}, ensure_ascii=False), flush=True)
    print(f"Report: {destination}", flush=True)
    return 0 if report["transport_and_rewarm_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
