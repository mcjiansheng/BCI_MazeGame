import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bci_maze.lk_mini.source import EEGChunk, EEGSource
from scripts import validate_lk_connection as validation


def successful_report():
    return {
        "sample_rate": 10, "samples": 140, "started_monotonic": 100.0,
        "acquisition_elapsed_seconds": 20.0,
        "capture_completed": True, "capture_error": "", "finite_samples": True,
        "finite_received_timestamps": True, "within_segment_continuity_errors": 0,
        "first_data_seconds": 0.5, "tail_silence_seconds": 0.1,
        "final_connection_state": "streaming",
        "segments": [
            {"segment_id": 0, "samples": 40, "sample_seconds": 4.0,
             "first_data_elapsed": 0.5, "last_data_elapsed": 4.9,
             "first_sample_monotonic": 100.5, "last_sample_monotonic": 104.4},
            {"segment_id": 1, "samples": 100, "sample_seconds": 10.0,
             "first_data_elapsed": 10.1, "last_data_elapsed": 19.9,
             "first_sample_monotonic": 110.0, "last_sample_monotonic": 119.9},
        ],
        "planned_injections": [{"type": "device_stream_stop", "scheduled_elapsed": 5.0}],
        "validation_events": [
            {"type": "device_stream_stop", "elapsed": 5.0, "scheduled_elapsed": 5.0,
             "completed_elapsed": 5.1, "http_status": 200,
             "state_before": "streaming", "segment_id_before": 0},
            {"type": "segment", "segment_id": 1, "elapsed": 8.0},
        ],
        "recovery_events": [
            {"type": "disconnected", "segment_id": 1, "time_monotonic": 108.0},
            {"type": "connected", "segment_id": 1, "time_monotonic": 109.8},
            {"type": "recovered", "segment_id": 1, "time_monotonic": 110.0},
        ],
    }


def test_summary_measures_fault_detection_and_first_data_separately_without_mutating():
    report = successful_report()
    original = copy.deepcopy(report)
    result = validation.evaluate_connection_report(report)
    assert report == original
    assert result["validation_passed"] and result["tail_active"]
    assert result["clock_alignment"] == "recorded"
    injection = result["injection_results"][0]
    assert injection["fault_to_detection_seconds"] == 3
    assert injection["detection_to_recovered_seconds"] == 2
    assert injection["fault_to_first_new_data_seconds"] == pytest.approx(5.1)
    assert injection["last_old_to_first_new_delivery_seconds"] == pytest.approx(5.2)
    assert injection["sample_clock_gap_seconds"] == pytest.approx(5.5)


@pytest.mark.parametrize("updates,expected", [
    ({"tail_silence_seconds": 12}, "stream_inactive_at_end"),
    ({"final_connection_state": "reconnecting"}, "stream_inactive_at_end"),
    ({"finite_samples": False}, "nonfinite_samples"),
    ({"finite_received_timestamps": False}, "nonfinite_received_timestamps"),
    ({"capture_completed": False}, "capture_incomplete"),
])
def test_finished_capture_requires_valid_samples_and_active_tail(updates, expected):
    report = successful_report()
    report.update(updates)
    result = validation.evaluate_connection_report(report)
    assert not result["validation_passed"]
    assert expected in result["validation_failures"]


def test_sdk_ready_without_new_samples_does_not_pass_recovery():
    report = successful_report()
    report["recovery_events"].pop()
    result = validation.evaluate_connection_report(report)
    assert not result["validation_passed"]
    assert "no_recovered_data" in result["injection_results"][0]["failures"]


def test_long_initial_outage_and_persistent_low_throughput_fail_despite_active_tail():
    report = successful_report()
    report.update(first_data_seconds=290, acquisition_elapsed_seconds=300,
                  samples=100, validation_events=[], planned_injections=[])
    result = validation.evaluate_connection_report(report)
    assert "initial_data_deadline_exceeded" in result["validation_failures"]
    assert "insufficient_expected_sample_coverage" in result["validation_failures"]
    report.update(first_data_seconds=1, samples=1000)
    result = validation.evaluate_connection_report(report)
    assert "insufficient_expected_sample_coverage" in result["validation_failures"]


def test_failed_http_injection_cannot_be_rescued_by_an_unrelated_recovery():
    report = successful_report()
    report["validation_events"][0].update(http_status=500, error="HTTPError")
    result = validation.evaluate_connection_report(report)
    assert not result["validation_passed"]
    assert "injection_failed" in result["injection_results"][0]["failures"]


def test_every_planned_injection_must_execute_and_receive_its_own_recovery():
    report = successful_report()
    second = {"type": "device_stream_stop", "scheduled_elapsed": 15.0}
    report["planned_injections"].append(second)
    result = validation.evaluate_connection_report(report)
    assert "injection_not_executed:device_stream_stop:15" in result["validation_failures"]
    report["validation_events"].append({**second, "elapsed": 15.0, "http_status": 200})
    result = validation.evaluate_connection_report(report)
    assert result["injection_results"][0]["passed"]
    assert not result["injection_results"][1]["passed"]


def test_one_packet_is_insufficient_to_demonstrate_resumed_sampling():
    report = successful_report()
    report["segments"][1].update(samples=1, sample_seconds=0.1)
    result = validation.evaluate_connection_report(report)
    assert "insufficient_recovered_samples" in result["injection_results"][0]["failures"]


def test_recovery_must_advance_the_segment_and_follow_the_injection():
    report = successful_report()
    report["validation_events"][0]["segment_id_before"] = 1
    report["segments"][1]["first_data_elapsed"] = 4.0
    result = validation.evaluate_connection_report(report)
    assert "segment_did_not_advance" in result["injection_results"][0]["failures"]
    assert "new_segment_data_predates_injection" in result["injection_results"][0]["failures"]


def test_legacy_clock_estimate_is_explicit_and_preserves_exact_recovery_stage():
    report = successful_report()
    del report["started_monotonic"]
    del report["segments"][1]["first_data_elapsed"]
    result = validation.evaluate_connection_report(report)
    assert result["validation_passed"]
    assert result["clock_alignment"] == "estimated_from_segment_notification"
    assert result["injection_results"][0]["first_new_data_basis"] == "recovered_event"
    assert result["injection_results"][0]["detection_to_recovered_seconds"] == 2


def test_software_restart_requires_forced_reconnect_event():
    report = successful_report()
    report["planned_injections"][0]["type"] = "requested_restart"
    report["validation_events"][0]["type"] = "requested_restart"
    assert not validation.evaluate_connection_report(report)["validation_passed"]
    report["recovery_events"][0]["type"] = "forced_reconnect"
    assert validation.evaluate_connection_report(report)["validation_passed"]


def test_saved_arrays_detect_a_single_missing_sample_and_reversed_segment_boundary():
    report = successful_report()
    report["samples"] = 4
    result = validation.evaluate_connection_report(
        report, sample_timestamps=[100, 100.1, 100.3, 100.4], segment_ids=[0, 0, 0, 0],
    )
    assert result["verified_within_segment_continuity_errors"] == 1
    result = validation.evaluate_connection_report(
        report, sample_timestamps=[100, 100.1, 99, 99.1], segment_ids=[0, 0, 1, 1],
    )
    assert result["segment_boundary_errors"] == 1
    assert not result["validation_passed"]


def test_cli_returns_failure_when_it_receives_initial_data_then_stays_silent(monkeypatch, tmp_path):
    clock = [100.0]

    class SilentAfterFirstChunk(EEGSource):
        sample_rate = 10
        channel_names = ("C3",)

        def start(self):
            self.sent = False

        def stop(self):
            pass

        def read(self, timeout=0.25):
            clock[0] += timeout
            if not self.sent:
                self.sent = True
                return EEGChunk(np.ones((1, 2)), np.array([100.0, 100.1]))
            return EEGChunk(np.empty((1, 0)), np.empty(0))

    monkeypatch.setattr(validation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    captured = {}

    def source_factory(backend, **parameters):
        captured.update(parameters)
        return SilentAfterFirstChunk()

    monkeypatch.setattr(validation, "create_source", source_factory)
    monkeypatch.setattr(sys, "argv", ["validate", "--duration", "3", "--output-dir", str(tmp_path)])
    assert validation.main() == 3
    report = json.loads((tmp_path / "connection_report.json").read_text(encoding="utf-8"))
    assert report["capture_completed"] and report["samples"] == 2
    assert "stream_inactive_at_end" in report["validation_failures"]
    assert report["connection_parameters"] == captured == {
        "auto_reconnect": True, "device_timeout": 5, "connect_timeout": 12.0,
        "no_data_timeout": 3.0, "shutdown_timeout": 4.0, "retry_delay": 0.5, "max_retry_delay": 3.0,
    }


@pytest.mark.parametrize("single_session", [False, True])
def test_cli_applies_and_records_selected_recovery_parameters(monkeypatch, tmp_path, single_session):
    clock = [100.0]
    captured = {}

    class EmptySource(EEGSource):
        sample_rate = 250
        channel_names = ("C3",)

        def start(self):
            pass

        def stop(self):
            pass

        def read(self, timeout=0.25):
            clock[0] += timeout
            return EEGChunk(np.empty((1, 0)), np.empty(0))

    def recovered_source(backend, **parameters):
        assert not single_session
        assert backend == "brainflow"
        captured.update(parameters)
        return EmptySource()

    def single_source(**parameters):
        assert single_session
        captured.update(parameters)
        return EmptySource()

    monkeypatch.setattr(validation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(validation, "create_source", recovered_source)
    monkeypatch.setattr(validation, "BrainFlowSource", single_source)
    argv = ["validate", "--duration", "0.5", "--output-dir", str(tmp_path),
            "--device-timeout", "2", "--connect-timeout", "8", "--no-data-timeout", "1.5",
            "--shutdown-timeout", "1", "--retry-delay", "0.2", "--max-retry-delay", "1"]
    if single_session:
        argv.append("--single-session")
    monkeypatch.setattr(sys, "argv", argv)
    assert validation.main() == 3
    report = json.loads((tmp_path / "connection_report.json").read_text(encoding="utf-8"))
    if single_session:
        assert captured == {"timeout": 2}
        assert report["connection_parameters"] == {"auto_reconnect": False, "device_timeout": 2}
    else:
        assert report["connection_parameters"] == captured == {
            "auto_reconnect": True, "device_timeout": 2, "connect_timeout": 8.0,
            "no_data_timeout": 1.5, "shutdown_timeout": 1.0, "retry_delay": 0.2, "max_retry_delay": 1.0,
        }


@pytest.mark.parametrize("flags", [
    ["--device-timeout", "0"], ["--device-timeout", "-1"], ["--device-timeout", "1.5"],
    ["--device-timeout", "600"],
    ["--connect-timeout", "0"], ["--connect-timeout", "nan"],
    ["--no-data-timeout", "-1"], ["--no-data-timeout", "inf"],
    ["--shutdown-timeout", "0"], ["--shutdown-timeout", "nan"],
    ["--retry-delay", "-0.1"], ["--retry-delay", "inf"],
    ["--max-retry-delay", "nan"], ["--retry-delay", "1", "--max-retry-delay", "0.5"],
    ["--duration", "nan"],
])
def test_invalid_recovery_parameters_fail_before_opening_source(monkeypatch, tmp_path, flags):
    def forbidden_source(*args, **kwargs):
        raise AssertionError("Validation must happen before opening a device")

    monkeypatch.setattr(validation, "create_source", forbidden_source)
    monkeypatch.setattr(validation, "BrainFlowSource", forbidden_source)
    monkeypatch.setattr(sys, "argv", ["validate", "--output-dir", str(tmp_path), *flags])
    with pytest.raises(SystemExit) as stopped:
        validation.main()
    assert stopped.value.code == 2
