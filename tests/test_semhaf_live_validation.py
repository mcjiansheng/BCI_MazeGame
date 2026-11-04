from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bci_maze.lk_mini.source import EEGChunk
from bci_maze.lk_mini.transport_log import TransportLogger
from scripts.validate_semhaf_live import summarize_live, read_new_records, terminate_owned_process


def evidence():
    events = [{"type": "disconnected", "segment_id": 1, "time_monotonic": 8.0},
              {"type": "recovered", "segment_id": 1, "time_monotonic": 10.0}]
    transport = [{"event": "start", "monotonic": 0.0, "sample_rate": 250, "window_samples": 1000},
                 {"event": "state", "monotonic": 8.0, "recovering": True, "recovery_events": events[:1]}]
    for segment, now, first in [(0, 4.0, 0.0), (1, 10.0, 6.0), (1, 19.5, 10.0)]:
        transport.append({"event": "chunk", "monotonic": now, "segment_id": segment, "sample_count": 1000,
                          "timestamp_start": first, "timestamp_end": first + 3.996, "continuity_errors": 0})
    transport.extend([{"event": "state", "monotonic": 10.0, "recovering": False, "recovery_events": events},
                      {"event": "summary", "capture_end_monotonic": 20.0, "acquisition_elapsed": 20.0, "tail_silence_seconds": .5, "error": "", "recovery_events": events}])
    windows = [{"monotonic": now + .01, "segment_id": segment, "window_samples": 1000,
                "segment_samples_received": 1000, "window_start": first, "window_end": first + 3.996,
                "window_span_seconds": 4.0, "reason": "bad_signal", "accepted": False, "inference_invoked": False}
               for segment, now, first in [(0, 4.0, 0.0), (1, 10.0, 6.0)]]
    injections = [{"request_started_monotonic": 5.0, "segment_before": 0, "http_status": 200}]
    return transport, windows, injections


def test_live_summary_requires_http_events_fresh_window_and_zero_gap_acceptance():
    transport, windows, injections = evidence()
    kwargs = dict(requested_duration=20, returncode=0, wear_state="unworn", expected_injection_count=1)
    report = summarize_live(transport, windows, injections, **kwargs)
    assert report["transport_and_rewarm_passed"]
    assert report["quality_gate_count"] == 2
    assert report["inference_invoked_count"] == 0
    assert report["MI_accuracy"] is None
    for change in ("http", "events", "window", "accepted", "finite", "missing", "baseline"):
        t, w, i = deepcopy(transport), deepcopy(windows), deepcopy(injections)
        if change == "http":
            i[0]["error"] = "timeout"
        elif change == "events":
            for row in t:
                row["recovery_events"] = []
        elif change == "window":
            w[1]["window_start"] = 0.0
        elif change == "accepted":
            w.append({**w[0], "monotonic": 9.0, "accepted": True})
        elif change == "finite":
            t[2]["nonfinite_data_values"] = 1
        elif change == "missing":
            i.clear()
        else:
            w.pop(0)
        assert not summarize_live(t, w, i, **kwargs)["transport_and_rewarm_passed"], change


def test_transport_logger_counts_nonfinite_and_closes_when_summary_write_fails(tmp_path, monkeypatch):
    logger = TransportLogger(tmp_path / "transport.jsonl")
    logger.start(0, 250, ["C3", "C4"], 1000)
    source = SimpleNamespace(recovering=False, connection_state="streaming", recovery_stats={}, recovery_events=[])
    logger.observe(EEGChunk(np.array([[np.nan, 2.0], [1.0, 2.0]]), np.array([1.0, 1.004])), source, 1.1)
    assert logger.nonfinite_data_values == 1
    file = logger.file
    monkeypatch.setattr(logger, "write", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        logger.finish(source, capture_end=2.0)
    assert file.closed and logger.file is None


def test_tail_reader_defers_partial_json_line(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"event":"start"}\n{"event":', encoding="utf-8")
    records, offset = read_new_records(path)
    assert records == [{"event": "start"}]
    with path.open("a", encoding="utf-8") as output:
        output.write('"chunk"}\n')
    records, _ = read_new_records(path, offset)
    assert records == [{"event": "chunk"}]


def test_abnormal_cleanup_targets_only_owned_windows_process_tree(monkeypatch):
    import scripts.validate_semhaf_live as live
    if live.os.name != "nt":
        pytest.skip("Windows process-tree cleanup")
    commands = []
    process = SimpleNamespace(pid=123456, poll=lambda: None, wait=lambda timeout: 1)
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="terminated", stderr="")
    monkeypatch.setattr(live.subprocess, "run", run)
    result = terminate_owned_process(process)
    assert commands == [["taskkill", "/PID", "123456", "/T", "/F"]]
    assert result["action"] == "terminate_owned_process_tree"
