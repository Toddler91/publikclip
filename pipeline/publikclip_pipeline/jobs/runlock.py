"""Per-job run lock: exactly one pipeline process may own a job.

Two processes on one job is not a theoretical race — it happens routinely.
Restarting the app orphans its pipeline child (reparented to pid 1, still
transcribing), and pressing Resume then starts a *second* run of the same
job. Both write the same checkpoint files, so the slower one silently
overwrites the faster one's work and the CPU cost doubles.

The lock is a file in the job dir, not a DB row, to match the project's rule
that artifacts on disk are the truth (PLAN.md §3):

    <job_dir>/run.lock   {"pid": ..., "started_at": ..., "stage": ...}

A lock whose pid is gone is stale and gets taken over — a killed process must
never wedge a job permanently. Liveness is `kill(pid, 0)`, which also reports
a SIGSTOP-suspended process as alive: a paused job is still owned.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

LOCK_NAME = "run.lock"


class JobBusyError(Exception):
    """Another live process already owns this job."""


def lock_path(job_dir: Path) -> Path:
    return job_dir / LOCK_NAME


def pid_alive(pid: int) -> bool:
    """Signal 0 probes existence without delivering anything. EPERM means the
    pid exists but belongs to someone else — still alive for our purposes."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def pid_paused(pid: int) -> bool:
    """True when the process is SIGSTOP-suspended. Unix-only; `ps` state T.

    There is no portable way to ask this, and no SIGSTOP on Windows at all,
    so a failed probe answers 'not paused' rather than raising.
    """
    if platform.system() == "Windows" or not pid_alive(pid):
        return False
    try:
        out = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return out[:1] == "T"


@dataclass
class LockInfo:
    pid: int
    started_at: float
    stage: str | None = None

    @property
    def alive(self) -> bool:
        return pid_alive(self.pid)

    @property
    def paused(self) -> bool:
        return pid_paused(self.pid)


def read(job_dir: Path) -> LockInfo | None:
    """The lock as written, or None if absent/corrupt. Does not check liveness."""
    try:
        raw = json.loads(lock_path(job_dir).read_text())
        return LockInfo(
            pid=int(raw["pid"]),
            started_at=float(raw.get("started_at", 0.0)),
            stage=raw.get("stage"),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def owner(job_dir: Path) -> LockInfo | None:
    """The live owner of this job, or None. Clears the lock if it is stale."""
    info = read(job_dir)
    if info is None:
        return None
    if info.alive:
        return info
    release(job_dir, pid=info.pid)  # stale: the owner died without cleaning up
    return None


def _write(job_dir: Path, info: LockInfo) -> None:
    path = lock_path(job_dir)
    tmp = path.with_suffix(".lock.tmp")
    tmp.write_text(json.dumps(
        {"pid": info.pid, "started_at": info.started_at, "stage": info.stage}
    ))
    tmp.replace(path)


def touch_stage(job_dir: Path, stage: str) -> None:
    """Record which stage the owner is in, for anything reporting job state."""
    info = read(job_dir)
    if info is None or info.pid != os.getpid():
        return
    info.stage = stage
    try:
        _write(job_dir, info)
    except OSError:
        pass  # bookkeeping only; never fail a stage over it


def release(job_dir: Path, pid: int | None = None) -> None:
    """Remove the lock. With `pid`, only if that pid still owns it — so a
    process that lost the lock to a takeover cannot delete the new owner's."""
    info = read(job_dir)
    if info is None:
        return
    if pid is not None and info.pid != pid:
        return
    try:
        lock_path(job_dir).unlink()
    except OSError:
        pass


class hold:
    """Context manager taking the lock for this process.

    Raises JobBusyError if a live process already holds it; the caller should
    surface that rather than starting a duplicate run.
    """

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.pid = os.getpid()

    def __enter__(self) -> "hold":
        self.job_dir.mkdir(parents=True, exist_ok=True)
        existing = owner(self.job_dir)
        if existing is not None:
            # Not re-entrant on purpose: a nested hold's __exit__ would drop
            # the lock while the outer one still believed it held it.
            if existing.pid == self.pid:
                raise JobBusyError(
                    "This job is already locked by this process — run_stages "
                    "cannot be nested."
                )
            state = "paused" if existing.paused else "running"
            raise JobBusyError(
                f"This job is already {state} in another process (pid "
                f"{existing.pid}). Stop that one first, or wait for it to finish."
            )
        _write(self.job_dir, LockInfo(pid=self.pid, started_at=time.time()))
        return self

    def __exit__(self, *exc) -> None:
        release(self.job_dir, pid=self.pid)
