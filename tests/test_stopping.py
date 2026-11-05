import signal
import threading

import pytest

from bci_maze.stopping import defer_stop


def test_nested_stop_completes_commits_and_restores_handler():
    old = signal.getsignal(signal.SIGINT)
    events = []
    def handler(*args):
        events.append("stop")
    signal.signal(signal.SIGINT, handler)
    try:
        with defer_stop():
            events.append("begin")
            with defer_stop():
                signal.raise_signal(signal.SIGINT)
                signal.raise_signal(signal.SIGINT)
                events.append("inner_commit")
            events.append("outer_commit")
        assert events == ["begin", "inner_commit", "outer_commit", "stop"]
        assert signal.getsignal(signal.SIGINT) is handler
    finally:
        signal.signal(signal.SIGINT, old)


def test_stop_inside_handled_exception_is_still_delivered():
    with pytest.raises(KeyboardInterrupt):
        try:
            raise ValueError("handled")
        except ValueError:
            with defer_stop():
                signal.raise_signal(signal.SIGINT)


def test_critical_operation_error_preserves_diagnostic():
    old = signal.getsignal(signal.SIGINT)
    with pytest.raises(OSError, match="disk full"):
        with defer_stop():
            signal.raise_signal(signal.SIGINT)
            raise OSError("disk full")
    assert signal.getsignal(signal.SIGINT) is old


def test_background_cleanup_does_not_replace_main_handler():
    old = signal.getsignal(signal.SIGINT)
    seen = []
    def cleanup():
        with defer_stop():
            seen.append(signal.getsignal(signal.SIGINT))
    thread = threading.Thread(target=cleanup)
    thread.start()
    thread.join()
    assert seen == [old]
