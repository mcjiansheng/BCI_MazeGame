"""Measure a live LK-Mini connection and optional deliberate session restarts."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bci_maze.lk_mini.source import BrainFlowSource, create_source


def evaluate_connection_report(
    report: dict, *, sample_timestamps=None, segment_ids=None,
    tail_silence_limit: float = 1.0, minimum_recovered_sample_seconds: float = 1.0,
    maximum_initial_data_seconds: float = 15.0, minimum_expected_sample_coverage: float = 0.9,
) -> dict:
    """Evaluate a saved capture without opening hardware or changing its report.

    Older reports lack the monotonic origin. Their recovery-stage durations remain
    exact; injection-relative durations use the observed segment notification as
    an explicitly estimated clock alignment.
    """
    failures = []
    events = report.get("validation_events", [])
    recovery = report.get("recovery_events", [])
    segments = {int(item["segment_id"]): dict(item) for item in report.get("segments", [])}
    rate = float(report["sample_rate"])
    origin = report.get("started_monotonic")
    clock_alignment = "recorded" if origin is not None else "unavailable"
    if origin is None:
        offsets = []
        for event in events:
            if event.get("type") != "segment":
                continue
            matches = [item for item in recovery if item.get("type") in ("disconnected", "forced_reconnect")
                       and item.get("segment_id") == event.get("segment_id")]
            if matches:
                offsets.append(matches[0]["time_monotonic"] - event["elapsed"])
        if offsets:
            origin = float(np.median(offsets))
            clock_alignment = "estimated_from_segment_notification"

    def elapsed(event):
        return None if origin is None else float(event["time_monotonic"] - origin)

    continuity_errors = int(report.get("within_segment_continuity_errors", 0))
    boundary_errors = 0
    if sample_timestamps is not None or segment_ids is not None:
        times = np.asarray(sample_timestamps, dtype=np.float64)
        ids = np.asarray(segment_ids)
        if times.ndim != 1 or ids.ndim != 1 or times.size != ids.size or not np.isfinite(times).all():
            failures.append("invalid_sample_timestamps_or_segment_ids")
        elif not np.isfinite(ids).all() or np.any(ids != np.floor(ids)):
            failures.append("invalid_segment_ids")
        else:
            if times.size != int(report.get("samples", 0)):
                failures.append("sample_count_mismatch")
            continuity_errors = 0
            transitions = np.diff(ids)
            boundary_errors = int(np.count_nonzero(transitions < 0))
            boundary_errors += int(np.count_nonzero((transitions != 0) & (np.diff(times) <= 0)))
            for segment in np.unique(ids):
                values = times[ids == segment]
                bad = int(np.count_nonzero(~np.isclose(np.diff(values), 1 / rate, rtol=1e-3, atol=1e-7)))
                continuity_errors += bad
                item = segments.setdefault(int(segment), {"segment_id": int(segment)})
                item.update(samples=int(values.size), sample_seconds=float(values.size / rate),
                            first_sample_monotonic=float(values[0]), last_sample_monotonic=float(values[-1]),
                            continuity_errors=bad)
    if not report.get("capture_completed"):
        failures.append("capture_incomplete")
    if report.get("capture_error"):
        failures.append("capture_error")
    if not report.get("samples", 0):
        failures.append("no_samples")
    if not report.get("finite_samples", False):
        failures.append("nonfinite_samples")
    if not report.get("finite_received_timestamps", True):
        failures.append("nonfinite_received_timestamps")
    if continuity_errors:
        failures.append("within_segment_discontinuity")
    if boundary_errors:
        failures.append("invalid_segment_boundary")
    tail_silence = float(report.get("tail_silence_seconds", float("inf")))
    tail_active = bool(report.get("samples", 0) and np.isfinite(tail_silence)
                       and 0 <= tail_silence <= tail_silence_limit)
    final_state = report.get("final_connection_state")
    if final_state is not None and final_state != "streaming":
        tail_active = False
    if not tail_active:
        failures.append("stream_inactive_at_end")

    injections = [event for event in events if event.get("type") in ("device_stream_stop", "requested_restart")]
    planned = report.get("planned_injections")
    if planned is not None:
        remaining = list(injections)
        for plan in planned:
            match = next((event for event in remaining if event["type"] == plan["type"]
                          and event.get("scheduled_elapsed") == plan["scheduled_elapsed"]), None)
            if match is None:
                failures.append(f"injection_not_executed:{plan['type']}:{plan['scheduled_elapsed']:g}")
            else:
                remaining.remove(match)
    injection_results = []
    for index, injection in enumerate(injections):
        errors = []
        at = float(injection["elapsed"])
        until = float(injections[index + 1]["elapsed"]) if index + 1 < len(injections) else float("inf")
        if injection.get("error") or (injection["type"] == "device_stream_stop"
                                     and not 200 <= int(injection.get("http_status", 0)) < 300):
            errors.append("injection_failed")
        if report.get("first_data_seconds") is None or float(report["first_data_seconds"]) > at:
            errors.append("no_data_before_injection")
        if injection.get("state_before", "streaming") != "streaming":
            errors.append("stream_not_active_before_injection")
        expected_type = "forced_reconnect" if injection["type"] == "requested_restart" else "disconnected"
        detection = next((event for event in recovery if event.get("type") == expected_type
                          and elapsed(event) is not None and at - 1e-6 <= elapsed(event) < until), None)
        recovered = None
        if detection is not None:
            recovered = next((event for event in recovery if event.get("type") == "recovered"
                              and event["time_monotonic"] >= detection["time_monotonic"]
                              and elapsed(event) < until), None)
        if detection is None:
            errors.append("no_detected_disconnection")
        if recovered is None:
            errors.append("no_recovered_data")
        segment = segments.get(int(recovered["segment_id"])) if recovered else None
        before_segment = injection.get("segment_id_before")
        if segment is None or not segment.get("samples", 0):
            errors.append("no_new_segment_samples")
        elif before_segment is not None and int(segment["segment_id"]) <= int(before_segment):
            errors.append("segment_did_not_advance")
        elif float(segment.get("sample_seconds", 0)) < minimum_recovered_sample_seconds:
            errors.append("insufficient_recovered_samples")
        first_data = segment.get("first_data_elapsed") if segment else None
        first_data_basis = "parent_delivery"
        if first_data is None and recovered is not None:
            first_data = elapsed(recovered)
            first_data_basis = "recovered_event"
        if first_data is not None and first_data < at:
            errors.append("new_segment_data_predates_injection")
        previous = [item for key, item in segments.items() if recovered and key < int(recovered["segment_id"])]
        previous = max(previous, key=lambda item: int(item["segment_id"])) if previous else None
        sample_gap = None
        if segment and previous and "first_sample_monotonic" in segment and "last_sample_monotonic" in previous:
            sample_gap = segment["first_sample_monotonic"] - previous["last_sample_monotonic"] - 1 / rate
            if sample_gap < -1e-6:
                errors.append("overlapping_segment_sample_clocks")
        item = dict(
            type=injection["type"], scheduled_elapsed=injection.get("scheduled_elapsed"),
            injection_elapsed=at, injection_completed_elapsed=injection.get("completed_elapsed"),
            http_status=injection.get("http_status"), passed=not errors, failures=errors,
            detected_elapsed=elapsed(detection) if detection else None,
            recovered_elapsed=elapsed(recovered) if recovered else None,
            first_new_data_elapsed=first_data, first_new_data_basis=first_data_basis,
            recovered_segment_id=int(recovered["segment_id"]) if recovered else None,
            fault_to_detection_seconds=None if detection is None else elapsed(detection) - at,
            detection_to_recovered_seconds=None if recovered is None else recovered["time_monotonic"] - detection["time_monotonic"],
            fault_to_first_new_data_seconds=None if first_data is None else first_data - at,
            new_segment_sample_seconds=None if segment is None else segment.get("sample_seconds"),
            sample_clock_gap_seconds=sample_gap,
        )
        if previous and first_data is not None and previous.get("last_data_elapsed") is not None:
            item["last_old_to_first_new_delivery_seconds"] = first_data - previous["last_data_elapsed"]
        injection_results.append(item)
        failures.extend(f"injection_{index + 1}:{error}" for error in errors)
    first_data = report.get("first_data_seconds")
    if first_data is not None and (not np.isfinite(first_data) or not 0 <= first_data <= maximum_initial_data_seconds):
        failures.append("initial_data_deadline_exceeded")
    measured_seconds = report.get("acquisition_elapsed_seconds")
    expected_sample_seconds = None
    sample_coverage = None
    if measured_seconds is not None:
        initial_wait = min(float(first_data or 0), maximum_initial_data_seconds)
        injected_outages = sum(max(0.0, item["fault_to_first_new_data_seconds"])
                               for item in injection_results if item["passed"])
        expected_sample_seconds = max(0.0, float(measured_seconds) - initial_wait - injected_outages)
        if expected_sample_seconds > 0:
            sample_coverage = float(report.get("samples", 0)) / rate / expected_sample_seconds
            if not np.isfinite(sample_coverage) or sample_coverage < minimum_expected_sample_coverage:
                failures.append("insufficient_expected_sample_coverage")
    return {
        "validation_passed": not failures, "validation_failures": failures,
        "validation_policy": {"tail_silence_limit_seconds": tail_silence_limit,
                              "minimum_recovered_sample_seconds": minimum_recovered_sample_seconds,
                              "maximum_initial_data_seconds": maximum_initial_data_seconds,
                              "minimum_expected_sample_coverage": minimum_expected_sample_coverage},
        "expected_sample_seconds": expected_sample_seconds,
        "expected_sample_coverage": sample_coverage,
        "tail_active": tail_active, "clock_alignment": clock_alignment,
        "injection_results": injection_results,
        "verified_within_segment_continuity_errors": continuity_errors,
        "segment_boundary_errors": boundary_errors,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument("--restart-at", default="", help="comma-separated elapsed seconds")
    parser.add_argument("--stop-stream-at", default="", help="stop device transmission at elapsed seconds to exercise no-data recovery")
    parser.add_argument("--wear-state", choices=("unworn", "worn", "unspecified"), default="unspecified")
    parser.add_argument("--device-timeout", type=int, default=5,
                        help="BrainFlow HTTP timeout, 1–599 seconds; applies to both source modes")
    recovery = parser.add_argument_group("automatic recovery (applies when --single-session is absent)")
    recovery.add_argument("--connect-timeout", type=float, default=12.0,
                          help="deadline for one BrainFlow connection attempt, seconds")
    recovery.add_argument("--no-data-timeout", type=float, default=3.0,
                          help="restart after this many seconds without valid EEG samples")
    recovery.add_argument("--shutdown-timeout", type=float, default=4.0,
                          help="grace period for the owned worker to release its session, seconds")
    recovery.add_argument("--retry-delay", type=float, default=0.5,
                          help="initial retry delay in seconds; may be zero")
    recovery.add_argument("--max-retry-delay", type=float, default=3.0,
                          help="maximum retry delay in seconds; at least --retry-delay")
    return parser


def connection_parameters(args: argparse.Namespace) -> dict:
    """Return the parameters actually applied to the selected source mode."""
    parameters = {"auto_reconnect": not args.single_session, "device_timeout": args.device_timeout}
    if not args.single_session:
        parameters.update({name: getattr(args, name) for name in (
            "connect_timeout", "no_data_timeout", "shutdown_timeout", "retry_delay", "max_retry_delay",
        )})
    return parameters


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    if not 1 <= args.device_timeout < 600:
        parser.error("--device-timeout must be between 1 and 599 seconds (BrainFlow HTTP range)")
    for name in ("device_timeout", "connect_timeout", "no_data_timeout", "shutdown_timeout"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not np.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("--retry-delay must be finite and non-negative")
    if not np.isfinite(args.max_retry_delay) or args.max_retry_delay < args.retry_delay:
        parser.error("--max-retry-delay must be finite and at least --retry-delay")
    restart_times = sorted(float(value) for value in args.restart_at.split(",") if value)
    stop_stream_times = sorted(float(value) for value in args.stop_stream_at.split(",") if value)
    planned_injections = sorted(
        [{"type": "requested_restart", "scheduled_elapsed": value} for value in restart_times]
        + [{"type": "device_stream_stop", "scheduled_elapsed": value} for value in stop_stream_times],
        key=lambda event: event["scheduled_elapsed"],
    )
    if not np.isfinite(args.duration) or args.duration <= 0 or any(not 0 < value < args.duration for value in restart_times + stop_stream_times):
        parser.error("duration must be positive and restart times must lie inside it")
    if args.single_session and (restart_times or stop_stream_times):
        parser.error("session restarts require the recovery source")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = connection_parameters(args)
    source = (BrainFlowSource(timeout=args.device_timeout) if args.single_session else
              create_source("brainflow", **parameters))
    data_parts, time_parts, receive_parts, segment_parts, delivery_parts = [], [], [], [], []
    events = []
    sample_count = 0
    last_received = None
    first_received = None
    max_read_gap = 0.0
    capture_error = ""
    current_segment = None
    next_report = 0.0
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    acquisition_elapsed = 0.0
    final_connection_state = "connecting"
    try:
        with source:
            while time.monotonic() - started < args.duration:
                now = time.monotonic()
                elapsed = now - started
                if restart_times and elapsed >= restart_times[0]:
                    requested_at = restart_times.pop(0)
                    injection = {"type": "requested_restart", "elapsed": time.monotonic() - started,
                                 "scheduled_elapsed": requested_at,
                                 "state_before": getattr(source, "connection_state", "streaming"),
                                 "segment_id_before": int(getattr(source, "segment_id", 0))}
                    source.request_reconnect("validation_requested_restart")
                    injection["completed_elapsed"] = time.monotonic() - started
                    events.append(injection)
                if stop_stream_times and elapsed >= stop_stream_times[0]:
                    scheduled = stop_stream_times.pop(0)
                    injection = {"type": "device_stream_stop", "elapsed": time.monotonic() - started,
                                 "scheduled_elapsed": scheduled,
                                 "state_before": getattr(source, "connection_state", "streaming"),
                                 "segment_id_before": int(getattr(source, "segment_id", 0))}
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    try:
                        with opener.open(f"http://{source.ip_address}/stream/stop", timeout=2) as response:
                            injection["http_status"] = response.status
                            response.read()
                    except Exception as exc:
                        injection["error"] = f"{type(exc).__name__}: {exc}"
                    injection["completed_elapsed"] = time.monotonic() - started
                    events.append(injection)
                    print(json.dumps(injection), flush=True)
                chunk = source.read(timeout=min(0.25, max(0.0, args.duration - (time.monotonic() - started))))
                now = time.monotonic()
                segment = int(getattr(chunk, "segment_id", 0))
                if current_segment != segment:
                    events.append({"type": "segment", "segment_id": segment,
                                   "elapsed": now - started})
                    current_segment = segment
                if chunk.data.shape[1]:
                    if last_received is not None:
                        max_read_gap = max(max_read_gap, now - last_received)
                    if first_received is None:
                        first_received = now
                    last_received = now
                    sample_count += chunk.data.shape[1]
                    data_parts.append(chunk.data)
                    time_parts.append(chunk.monotonic_timestamps)
                    receive_parts.append(chunk.received_timestamps if chunk.received_timestamps is not None
                                         else chunk.monotonic_timestamps)
                    segment_parts.append(np.full(chunk.data.shape[1], segment, dtype=np.int64))
                    delivery_parts.append(np.full(chunk.data.shape[1], now, dtype=np.float64))
                if now - started >= next_report:
                    status = {"elapsed": round(now - started, 2), "samples": sample_count,
                              "segment_id": segment,
                              "state": getattr(source, "connection_state", "streaming"),
                              "silence_seconds": round(now - (last_received or started), 3)}
                    print(json.dumps(status), flush=True)
                    events.append({"type": "progress", **status})
                    next_report = now - started + 5
            acquisition_elapsed = time.monotonic() - started
            final_connection_state = getattr(source, "connection_state", "streaming")
    except KeyboardInterrupt:
        capture_error = "KeyboardInterrupt"
    except Exception as exc:
        capture_error = f"{type(exc).__name__}: {exc}"
        print(capture_error, flush=True)
    finally:
        if not acquisition_elapsed:
            acquisition_elapsed = time.monotonic() - started
    source_stats = getattr(source, "recovery_stats", {})
    source_events = getattr(source, "recovery_events", [])
    samples = np.concatenate(data_parts, axis=1) if data_parts else np.empty((len(source.channel_names), 0))
    timestamps = np.concatenate(time_parts) if time_parts else np.empty(0)
    segments = np.concatenate(segment_parts) if segment_parts else np.empty(0, dtype=np.int64)
    delivered = np.concatenate(delivery_parts) if delivery_parts else np.empty(0)
    received = np.concatenate(receive_parts) if receive_parts else np.empty(0)
    segment_summaries = []
    continuity_errors = 0
    for segment in np.unique(segments):
        mask = segments == segment
        segment_times = timestamps[mask]
        deltas = np.diff(segment_times)
        bad = int(np.count_nonzero((deltas <= 0) | (deltas > 2.5 / source.sample_rate)))
        continuity_errors += bad
        segment_summaries.append({"segment_id": int(segment), "samples": int(mask.sum()),
                                  "sample_seconds": float(mask.sum() / source.sample_rate),
                                  "first_data_elapsed": float(delivered[mask][0] - started),
                                  "last_data_elapsed": float(delivered[mask][-1] - started),
                                  "first_sample_monotonic": float(segment_times[0]),
                                  "last_sample_monotonic": float(segment_times[-1]),
                                  "continuity_errors": bad})
    result = {
        "started_utc": started_utc,
        "started_monotonic": started,
        "requested_duration_seconds": args.duration,
        "acquisition_elapsed_seconds": acquisition_elapsed,
        "single_session": args.single_session,
        "connection_parameters": parameters,
        "wear_state": args.wear_state,
        "purpose": "transport continuity and session recovery",
        "channels": list(source.channel_names), "sample_rate": source.sample_rate,
        "samples": sample_count,
        "wall_sample_rate": sample_count / max(acquisition_elapsed, 1e-9),
        "first_data_seconds": None if first_received is None else first_received - started,
        "last_data_seconds": None if last_received is None else last_received - started,
        "max_read_gap_seconds": max_read_gap,
        "tail_silence_seconds": acquisition_elapsed if last_received is None else
            max(0.0, acquisition_elapsed - (last_received - started)),
        "capture_error": capture_error,
        "capture_completed": not capture_error and acquisition_elapsed >= args.duration,
        "finite_samples": bool(np.isfinite(samples).all()),
        "finite_received_timestamps": bool(np.isfinite(received).all()),
        "final_connection_state": final_connection_state,
        "within_segment_continuity_errors": continuity_errors,
        "segments": segment_summaries,
        "validation_events": events,
        "planned_injections": planned_injections,
        "recovery_stats": source_stats, "recovery_events": source_events,
        "restart_method": "software request_reconnect" if args.restart_at else None,
        "data_stop_method": "device HTTP /stream/stop" if args.stop_stream_at else None,
    }
    result.update(evaluate_connection_report(result, sample_timestamps=timestamps, segment_ids=segments))
    if sample_count:
        np.savez_compressed(args.output_dir / "capture.npz", data=samples.astype(np.float32),
                            timestamps=timestamps, received_timestamps=received, delivery_timestamps=delivered,
                            segment_ids=segments, sample_rate=np.asarray(source.sample_rate),
                            channel_names=np.asarray(source.channel_names))
    report = args.output_dir / "connection_report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report}", flush=True)
    print(json.dumps({key: result[key] for key in ("validation_passed", "validation_failures", "injection_results")}, ensure_ascii=False), flush=True)
    return 0 if result["validation_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
