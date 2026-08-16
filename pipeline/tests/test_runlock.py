"""Run lock: one process per job, and never a permanently wedged job."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from publikclip_pipeline.jobs import runlock


def test_hold_writes_and_releases(tmp_path):
    job = tmp_path / "job"
    assert runlock.owner(job) is None
    with runlock.hold(job):
        info = runlock.owner(job)
        assert info is not None and info.pid == os.getpid()
        assert runlock.lock_path(job).exists()
    assert runlock.owner(job) is None
    assert not runlock.lock_path(job).exists()


def test_nested_hold_refused(tmp_path):
    """Re-entrancy would let the inner exit drop the outer's lock."""
    job = tmp_path / "job"
    with runlock.hold(job):
        with pytest.raises(runlock.JobBusyError):
            with runlock.hold(job):
                pass
        assert runlock.owner(job).pid == os.getpid()  # outer still holds it


def test_sequential_holds_are_fine(tmp_path):
    job = tmp_path / "job"
    with runlock.hold(job):
        pass
    with runlock.hold(job):
        pass


def test_stale_lock_is_taken_over(tmp_path):
    """A killed owner must not wedge the job forever."""
    job = tmp_path / "job"
    job.mkdir()
    runlock.lock_path(job).write_text(json.dumps({"pid": 2**30, "started_at": 0.0}))
    assert runlock.owner(job) is None          # reported dead
    assert not runlock.lock_path(job).exists()  # and cleaned up
    with runlock.hold(job):
        assert runlock.owner(job).pid == os.getpid()


@pytest.fixture
def live_foreign_pid():
    """A pid that is running and is not ours.

    Spawned rather than hardcoded to 1: that is init on Unix, but Windows
    allocates pids in multiples of four starting at 0, so 1 is never a live
    process there and the test would pass for the wrong reason.
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_live_foreign_lock_blocks(tmp_path, live_foreign_pid):
    job = tmp_path / "job"
    job.mkdir()
    runlock.lock_path(job).write_text(
        json.dumps({"pid": live_foreign_pid, "started_at": 0.0})
    )
    with pytest.raises(runlock.JobBusyError, match="another process"):
        with runlock.hold(job):
            pass


def test_corrupt_lock_is_ignored(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    runlock.lock_path(job).write_text("{not json")
    assert runlock.owner(job) is None
    with runlock.hold(job):
        pass


def test_release_only_by_owner(tmp_path):
    """A process that lost the lock must not delete the new owner's."""
    job = tmp_path / "job"
    with runlock.hold(job):
        runlock.release(job, pid=os.getpid() + 12345)  # someone else's pid
        assert runlock.lock_path(job).exists()


def test_second_process_is_refused(tmp_path):
    """The real shape of the bug: two OS processes, one job dir."""
    job = tmp_path / "job"
    job.mkdir()
    script = textwrap.dedent("""
        import sys, time
        from pathlib import Path
        from publikclip_pipeline.jobs import runlock
        with runlock.hold(Path(sys.argv[1])):
            print("held", flush=True)
            time.sleep(30)
    """)
    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(job)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(runlock.JobBusyError, match="another process"):
            with runlock.hold(job):
                pass
    finally:
        holder.kill()
        holder.wait(timeout=10)

    # once that process is gone the lock is stale, not permanent
    assert runlock.owner(job) is None
    with runlock.hold(job):
        pass


def test_touch_stage_records_progress(tmp_path):
    job = tmp_path / "job"
    with runlock.hold(job):
        runlock.touch_stage(job, "asr")
        assert runlock.owner(job).stage == "asr"


def test_touch_stage_ignores_foreign_lock(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    runlock.lock_path(job).write_text(json.dumps({"pid": 1, "started_at": 0.0, "stage": None}))
    runlock.touch_stage(job, "asr")
    assert runlock.read(job).stage is None  # not ours to write
