"""Job queue + checkpoint/resume contract tests.

The resume guarantee is the whole point of M0: kill anywhere, re-run, and
only missing/stale work repeats. These tests exercise that contract without
any media."""

import json

import pytest

from publikclip_pipeline import config
from publikclip_pipeline.jobs import queue


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path / "home"))
    yield


@pytest.fixture(autouse=True)
def stub_ffmpeg(monkeypatch):
    """run_stages ensures an ffmpeg before any stage runs. These tests are
    about the queue and never touch media, so keep them off the resolution
    chain entirely — unstubbed, a machine with no ffmpeg would download one."""
    from publikclip_pipeline.render import ffmpeg_bin

    monkeypatch.setattr(ffmpeg_bin, "ensure_present", lambda progress=None: "/fake/ffmpeg")
    yield


def _settings_json() -> str:
    return json.dumps(config.Settings().to_json())


class CountingStage(queue.Stage):
    name = "counting"
    schema_version = 1

    def __init__(self):
        self.runs = 0

    def run(self, ctx):
        self.runs += 1
        return {"runs": self.runs}


class FailingStage(queue.Stage):
    name = "failing"
    schema_version = 1

    def run(self, ctx):
        raise queue.StageError("boom, but politely")


class ArtifactStage(queue.Stage):
    name = "artifact"
    schema_version = 1

    def __init__(self):
        self.runs = 0

    def run(self, ctx):
        self.runs += 1
        out = ctx.job_dir / "artifact.bin"
        out.write_bytes(b"data")
        return {"path": str(out)}

    def artifacts_ok(self, ctx, data):
        from pathlib import Path

        return Path(data["path"]).exists()


def _noop_progress(stage, fraction, message):
    pass


def test_create_and_get_job():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    fetched = queue.get_job(job.id)
    assert fetched is not None
    assert fetched.source == "/tmp/x.mp4"
    assert job.dir.exists()
    assert (job.dir / "settings.json").exists()


def test_stage_runs_once_then_caches():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = CountingStage()
    queue.run_stages(job, [stage], _noop_progress)
    queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 1  # second run served from checkpoint


def test_schema_version_bump_invalidates():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = CountingStage()
    queue.run_stages(job, [stage], _noop_progress)
    stage.schema_version = 2
    queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 2


def test_missing_artifact_invalidates_checkpoint():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = ArtifactStage()
    queue.run_stages(job, [stage], _noop_progress)
    (job.dir / "artifact.bin").unlink()
    queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 2


def test_corrupt_checkpoint_reruns():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = CountingStage()
    queue.run_stages(job, [stage], _noop_progress)
    queue.checkpoint_path(job, stage.name).write_text("{not json")
    queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 2


def test_stage_error_marks_job_failed():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    with pytest.raises(queue.StageError):
        queue.run_stages(job, [FailingStage()], _noop_progress)
    fetched = queue.get_job(job.id)
    assert fetched.status == "failed"
    assert "politely" in (fetched.error or "")


def test_failure_then_resume_skips_completed_stages():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    counting = CountingStage()
    with pytest.raises(queue.StageError):
        queue.run_stages(job, [counting, FailingStage()], _noop_progress)
    assert counting.runs == 1

    class FixedStage(queue.Stage):
        name = "failing"  # same name — simulates the bug being fixed
        schema_version = 1

        def run(self, ctx):
            return {"ok": True}

    results = queue.run_stages(job, [counting, FixedStage()], _noop_progress)
    assert counting.runs == 1  # not re-run
    assert results["failing"] == {"ok": True}
    assert queue.get_job(job.id).status == "done"


def test_ffmpeg_is_ensured_before_any_stage(monkeypatch):
    """Fetching, not merely resolving.

    A resumed job serves ingest from its checkpoint, so nothing else in the
    pass will obtain a binary. Only doing add_to_path() here leaves a machine
    with none on PATH running the whole pipeline without ffmpeg.
    """
    from publikclip_pipeline.render import ffmpeg_bin

    calls: list[str] = []

    def fake_ensure(progress=None):
        calls.append("ensure")
        return "/fake/ffmpeg"

    monkeypatch.setattr(ffmpeg_bin, "ensure_present", fake_ensure)
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    queue.run_stages(job, [CountingStage()], _noop_progress)
    assert calls == ["ensure"]


def test_missing_ffmpeg_fails_before_any_stage_runs(monkeypatch):
    """None on PATH, none on disk, none fetchable — say so up front, rather
    than several stages deep as a bare FileNotFoundError naming nothing."""
    from publikclip_pipeline.render import ffmpeg_bin

    monkeypatch.setattr(ffmpeg_bin, "ensure_present", lambda progress=None: None)
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = CountingStage()
    with pytest.raises(queue.StageError, match="ffmpeg"):
        queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 0
