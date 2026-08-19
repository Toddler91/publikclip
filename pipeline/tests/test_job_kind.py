"""Clip jobs and caption jobs are different things and list separately."""

import json
import sqlite3

import pytest

from publikclip_pipeline import config
from publikclip_pipeline.captions import tool as caption_tool
from publikclip_pipeline.jobs import queue


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path / "home"))
    yield


def _settings():
    return json.dumps(config.Settings().to_json())


def test_defaults_to_clip():
    assert queue.create_job("file", "/tmp/a.mp4", _settings()).kind == "clip"


def test_kind_round_trips():
    job = queue.create_job("file", "/tmp/a.mp4", _settings(), kind="caption")
    assert queue.get_job(job.id).kind == "caption"


def test_kind_is_written_beside_the_job():
    """The desktop shell builds its rail by scanning directories and never
    opens the database, so the kind has to be readable from disk."""
    job = queue.create_job("file", "/tmp/a.mp4", _settings(), kind="caption")
    assert (job.dir / queue.KIND_FILE).read_text(encoding="utf-8").strip() == "caption"


def test_bad_kind_is_refused():
    with pytest.raises(ValueError):
        queue.create_job("file", "/tmp/a.mp4", _settings(), kind="nonsense")


def test_database_without_the_column_is_migrated():
    """Older installs have a jobs table that CREATE TABLE IF NOT EXISTS will
    not alter, so the column is grafted on at connect time."""
    config.ensure_home()
    con = sqlite3.connect(config.db_path())
    con.executescript(
        "DROP TABLE IF EXISTS jobs;"
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, created_at REAL NOT NULL,"
        " source_type TEXT NOT NULL, source TEXT NOT NULL, title TEXT,"
        " status TEXT NOT NULL DEFAULT 'pending', error TEXT,"
        " settings_json TEXT NOT NULL);"
        "INSERT INTO jobs VALUES ('old', 0, 'file', '/tmp/x.mp4', NULL, 'done', NULL, '{}');"
    )
    con.commit()
    con.close()

    job = queue.get_job("old")
    assert job is not None and job.kind == "clip"


def test_captioning_does_not_adopt_a_clip_job(tmp_path):
    """A clip job for the same file has a transcript worth reusing, but taking
    its directory would retitle it and mark it done as though the clipping run
    had finished."""
    src = tmp_path / "v.mp4"
    src.write_bytes(b"")
    queue.create_job("file", str(src.resolve()), _settings(), kind="clip")
    assert caption_tool._find_existing_job(src) is None

    made = queue.create_job("file", str(src.resolve()), _settings(), kind="caption")
    assert caption_tool._find_existing_job(src).id == made.id
