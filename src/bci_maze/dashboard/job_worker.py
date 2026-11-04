"""Run one existing script and translate a local stop request to KeyboardInterrupt."""

from __future__ import annotations

import _thread
from pathlib import Path
import runpy
import signal
import sys
import threading


PROTECTED_FUNCTIONS = {
    "save_mi_recording", "save_synthetic_mi_recording", "save_model_bundle", "save_semhaf_bundle",
    "stop", "close", "__exit__", "finish",
}
_stop_delivered = threading.Event()
_deferred_notice = threading.Event()
_signal_lock = threading.RLock()


def stop_signal_handler(_signum, frame) -> None:
    """Complete a file write or resource release before honoring a late stop.

    Signal handlers execute on the main thread. Checking the live call stack
    avoids a race between a watcher thread observing a flag and signal delivery.
    The watcher retries deferred requests until the workflow finishes or one
    interrupt is delivered. A delivered interrupt is never repeated in cleanup.
    """
    current = frame
    while current is not None:
        if current.f_code.co_name in PROTECTED_FUNCTIONS:
            if not _deferred_notice.is_set():
                _deferred_notice.set()
                print("正在完成文件保存或资源清理，等待当前步骤结束。", flush=True)
            return
        current = current.f_back
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
    _deferred_notice.clear()
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
