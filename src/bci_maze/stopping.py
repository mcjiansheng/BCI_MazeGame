"""Explicit SIGINT deferral for state commits, persistence and owned cleanup."""

from contextlib import contextmanager
import signal
import threading


@contextmanager
def defer_stop():
    """Finish the main-thread critical section, then deliver one pending SIGINT.

    Nested sections forward the request to their enclosing handler. Background
    cleanup never changes the main thread's signal handling.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    pending = False

    def remember_stop(_signum, _frame):
        nonlocal pending
        pending = True

    signal.signal(signal.SIGINT, remember_stop)
    try:
        yield
    except BaseException:
        signal.signal(signal.SIGINT, previous)
        raise
    else:
        signal.signal(signal.SIGINT, previous)
        if pending:
            if callable(previous):
                previous(signal.SIGINT, None)
            elif previous == signal.SIG_DFL:
                signal.default_int_handler(signal.SIGINT, None)
