"""Run one existing script and translate a local stop request to KeyboardInterrupt."""

from __future__ import annotations

import _thread
from pathlib import Path
import runpy
import signal
import sys
import threading


_stop_delivered = threading.Event()
_signal_lock = threading.RLock()


def stop_signal_handler(_signum, frame) -> None:
    """Deliver a dashboard stop once; explicit critical sections defer SIGINT."""
    with _signal_lock:
        if _stop_delivered.is_set():
            return
        _stop_delivered.set()
        raise KeyboardInterrupt


def main() -> int:
    stop_path = Path(sys.argv[1])
    script = sys.argv[2]
    sys.argv = sys.argv[2:]
    finished = threading.Event()
    _stop_delivered.clear()
    signal.signal(signal.SIGINT, stop_signal_handler)

    def watch_stop() -> None:
        while not finished.wait(0.1):
            if stop_path.exists():
                print("控制台已请求停止，正在保存记录并释放资源。", flush=True)
                while not finished.is_set():
                    with _signal_lock:
                        if _stop_delivered.is_set():
                            return
                        _thread.interrupt_main()
                    if finished.wait(0.15):
                        return
                return

    watcher = threading.Thread(target=watch_stop, name="Dashboard cooperative stop", daemon=True)
    watcher.start()
    try:
        runpy.run_path(script, run_name="__main__")
    except KeyboardInterrupt:
        return 130
    finally:
        finished.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
