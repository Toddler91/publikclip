"""Deleting a job: rows and directory both, and never one without the other."""

import json

import pytest

from publikclip_pipeline import config
from publikclip_pipeline.jobs import queue, runlock


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path / "home"))
    yield


def _job():
    return queue.create_job("file", "/tmp/x.mp4", json.dumps(config.Settings().to_json()))


def test_deletes_row_and_directory():
    job = _job()
    (job.dir / "asr.json").write_text("{}")
    out = queue.delete_job(job.id)
    assert out["files_removed"] is True
    assert not job.dir.exists()
    assert queue.get_job(job.id) is None


def test_reports_the_space_it_freed():
    job = _job()
    (job.dir / "audio16k.wav").write_bytes(b"x" * 5000)
    assert queue.delete_job(job.id)["freed_bytes"] >= 5000


def test_stage_rows_go_too():
    """A surviving stage_runs row would resurrect as a phantom on a reused id."""
    job = _job()
    queue.mark_stage(job.id, "asr", "done", 1)
    assert queue.stage_statuses(job.id)
    queue.delete_job(job.id)
    assert queue.stage_statuses(job.id) == {}


def test_keep_files_forgets_the_job_but_leaves_the_data():
    job = _job()
    (job.dir / "asr.json").write_text("{}")
    out = queue.delete_job(job.id, remove_files=False)
    assert out["files_removed"] is False
    assert job.dir.exists()
    assert queue.get_job(job.id) is None


def test_unknown_job_raises():
    with pytest.raises(KeyError):
        queue.delete_job("nope")


def test_refuses_while_a_live_process_owns_it():
    """Deleting checkpoints under a running pipeline would surface as a failure
    far away from the cause."""
    job = _job()
    with runlock.hold(job.dir):
        with pytest.raises(runlock.JobBusyError, match="Stop it first"):
            queue.delete_job(job.id)
    assert queue.get_job(job.id) is not None  # still there
    queue.delete_job(job.id)  # and deletable once the lock is released


def test_a_stale_lock_does_not_block_deletion():
    """The whole point of deleting a dead job is that its owner is gone."""
    job = _job()
    runlock.lock_path(job.dir).write_text(json.dumps({"pid": 2**30, "started_at": 0.0}))
    queue.delete_job(job.id)
    assert queue.get_job(job.id) is None
