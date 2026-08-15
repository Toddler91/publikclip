"""Die when the app that started us is gone.

The pipeline is a sidecar: the Tauri shell spawns it and reads JSONL from its
stdout. If the shell exits, nothing tells this process — it keeps a whole
transcription running for an app nobody is looking at, and the next Resume
starts a *second* copy of the same job.

Watching getppid() does not work here: the immediate parent is the `uv`
wrapper, which survives the app and simply gets reparented to pid 1 along
with us. What does change is the pipe — when the shell dies, the read end of
our stdout closes, and writing to it fails (EPIPE). So we write a heartbeat
into that pipe on a timer and exit the moment it breaks.

The heartbeat doubles as liveness for the UI, which is why it carries a
timestamp rather than being a bare newline.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time

HEARTBEAT_SEC = 5.0
EXIT_CODE_PARENT_GONE = 3


def die_with_parent(interval: float = HEARTBEAT_SEC) -> None:
    """Start the heartbeat watchdog. Only meaningful with --jsonl, where
    stdout is a pipe to the app rather than a terminal."""
    # Default SIGPIPE handling kills the process silently on the first failed
    # write; Python sets it to SIG_IGN so we can see the error and exit clean.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (AttributeError, ValueError, OSError):
        pass  # no SIGPIPE on Windows, or not on the main thread

    def beat() -> None:
        while True:
            time.sleep(interval)
            payload = json.dumps({"event": "heartbeat", "t": round(time.time(), 3)})
            try:
                sys.stdout.write(payload + "\n")
                sys.stdout.flush()
            except (BrokenPipeError, ValueError, OSError):
                # The app is gone. os._exit skips atexit/finally, which would
                # otherwise try to flush to the same broken pipe — the run
                # lock is reclaimed by liveness check, so leaving it is safe.
                os._exit(EXIT_CODE_PARENT_GONE)

    threading.Thread(target=beat, name="parent-watchdog", daemon=True).start()
