"""Mini-EEG08 / Cyton-compatible connection diagnostic.

Usage:
    python scripts/test_mini_eeg_connection.py --port COM5 --seconds 10

Important:
- Close OpenBCI GUI before running this script; the serial port is normally exclusive.
- This script assumes the Mini-EEG08 is compatible with BrainFlow CYTON_BOARD over serial.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as serial_list_ports
    except ImportError:
        return []
    return [p.device for p in serial_list_ports.comports()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Windows serial port, e.g. COM5")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--output",
        default="outputs/mini_eeg08_connection_test.csv",
        help="BrainFlow raw CSV output path",
    )
    args = parser.parse_args()

    ports = list_ports()
    print("Detected serial ports:", ports or "(pyserial unavailable / no ports)")
    if not args.port:
        print("ERROR: pass --port COMx after identifying the Mini-EEG08 port.")
        return 2

    try:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        from brainflow.data_filter import DataFilter
    except ImportError:
        print(
            "ERROR: BrainFlow is not installed. Run "
            "'python -m pip install -r requirements.txt' first."
        )
        return 2

    board_id = BoardIds.CYTON_BOARD.value
    params = BrainFlowInputParams()
    params.serial_port = args.port
    params.timeout = 10

    BoardShim.enable_dev_board_logger()
    print("\nBrainFlow board descriptor:")
    print(BoardShim.get_board_descr(board_id))

    board = BoardShim(board_id, params)

    try:
        print(f"\nPreparing session on {args.port} ...")
        board.prepare_session()
        print("Session prepared.")

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        package_channel = BoardShim.get_package_num_channel(board_id)
        marker_channel = BoardShim.get_marker_channel(board_id)
        nominal_fs = BoardShim.get_sampling_rate(board_id)

        print(f"Nominal sampling rate from BrainFlow descriptor: {nominal_fs} Hz")
        print(f"EEG row indices: {eeg_channels}")
        print(f"Timestamp row: {timestamp_channel}")
        print(f"Package row: {package_channel}")
        print(f"Marker row: {marker_channel}")

        print("\nStarting stream ...")
        board.start_stream(45000, "")
        t0 = time.time()

        marker_sent = False
        while time.time() - t0 < args.seconds:
            elapsed = time.time() - t0
            if not marker_sent and elapsed >= args.seconds / 2:
                board.insert_marker(99.0)
                marker_sent = True
                print("Inserted marker 99.")
            time.sleep(0.05)

        board.stop_stream()
        data = board.get_board_data()
        print(f"\nRaw BrainFlow array shape: {data.shape}")

        if data.shape[1] == 0:
            print("FAIL: no samples received.")
            return 3

        eeg = data[eeg_channels, :]
        print(f"EEG shape: {eeg.shape}")
        print(f"Non-finite EEG values: {np.size(eeg) - np.isfinite(eeg).sum()}")

        print("\nPer-channel EEG statistics (BrainFlow units, normally uV for EXG):")
        for i, row in enumerate(eeg, start=1):
            finite = row[np.isfinite(row)]
            if finite.size:
                print(
                    f"  EEG{i}: mean={finite.mean():.3f}, "
                    f"std={finite.std():.3f}, "
                    f"min={finite.min():.3f}, max={finite.max():.3f}"
                )
            else:
                print(f"  EEG{i}: no finite samples")

        ts = data[timestamp_channel, :]
        finite_ts = ts[np.isfinite(ts)]
        if finite_ts.size >= 2:
            monotonic = bool(np.all(np.diff(finite_ts) >= 0))
            span = finite_ts[-1] - finite_ts[0]
            effective_fs = (finite_ts.size - 1) / span if span > 0 else float("nan")
            print(f"\nTimestamp monotonic: {monotonic}")
            print(f"Timestamp span: {span:.3f} s")
            print(f"Effective rate estimated from timestamps: {effective_fs:.2f} Hz")
        else:
            print("\nWARNING: insufficient timestamp data.")

        packages = data[package_channel, :]
        finite_pkg = packages[np.isfinite(packages)].astype(int)
        if finite_pkg.size >= 2:
            steps = np.diff(finite_pkg) % 256
            gap_like = int(np.count_nonzero(steps != 1))
            print(f"\nPackage counter first 20: {finite_pkg[:20].tolist()}")
            print(
                "Non-unit package-counter transitions (mod 256 heuristic): "
                f"{gap_like}"
            )
            print(
                "Note: this is a diagnostic heuristic; confirm Mini-EEG08 counter "
                "semantics with the vendor before interpreting every transition as packet loss."
            )

        markers = data[marker_channel, :]
        marker_positions = np.flatnonzero(np.isclose(markers, 99.0))
        print(f"\nMarker 99 positions: {marker_positions.tolist()}")
        if marker_positions.size == 0:
            print(
                "WARNING: marker 99 was not found. The compatible board may not "
                "implement BrainFlow markers exactly like Cyton, or marker handling "
                "needs vendor-specific configuration."
            )

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        DataFilter.write_file(data, str(output), "w")
        print(f"\nSaved BrainFlow raw CSV to: {output}")
        print("\nPASS: session opened and samples were received.")
        return 0

    except Exception as exc:
        print("\nFAIL:", repr(exc))
        print(
            "Check: correct COM port, OpenBCI GUI fully closed, device powered, "
            "and vendor confirmation that Mini-EEG08 uses BrainFlow CYTON_BOARD over serial."
        )
        return 1

    finally:
        try:
            if board.is_prepared():
                try:
                    board.stop_stream()
                except Exception:
                    pass
                board.release_session()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
