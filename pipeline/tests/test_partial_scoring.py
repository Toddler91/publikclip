"""Scoring keeps what it already paid for when the LLM runs out mid-list.

35 candidates means 35 LLM calls; a quota that dies at candidate 19 used to
discard all nineteen and fail the stage outright.
"""

import json

import pytest

from publikclip_pipeline.scoring import partial as partial_mod

CANDS = [{"start": float(i * 10), "end": float(i * 10 + 8)} for i in range(5)]


def _store(tmp_path, cands=None):
    return partial_mod.PartialScores.load_or_new(
        tmp_path / partial_mod.PARTIAL_NAME,
        partial_mod.candidates_fingerprint(cands or CANDS),
    )


def test_scores_round_trip(tmp_path):
    p = _store(tmp_path)
    p.entries[0] = {"start": 0.0, "summary": "first"}
    p.entries[2] = {"start": 20.0, "summary": "third"}
    p.save()

    again = _store(tmp_path)
    assert len(again) == 2
    assert sorted(again.entries) == [0, 2]
    assert again.entries[2]["summary"] == "third"


def test_ordered_follows_candidate_order(tmp_path):
    p = _store(tmp_path)
    for i in (3, 0, 1):
        p.entries[i] = {"summary": str(i)}
    assert [e["summary"] for e in p.ordered()] == ["0", "1", "3"]


def test_different_candidates_discard_scores(tmp_path):
    """Re-running the candidates stage must not pair old scores with new
    windows — the score belongs to a specific span, not an index."""
    p = _store(tmp_path)
    p.entries[0] = {"summary": "old"}
    p.save()

    moved = [{"start": 5.0, "end": 13.0}] + CANDS[1:]
    assert len(_store(tmp_path, moved)) == 0


def test_fewer_candidates_discard_scores(tmp_path):
    p = _store(tmp_path)
    p.entries[0] = {"summary": "old"}
    p.save()
    assert len(_store(tmp_path, CANDS[:3])) == 0


def test_corrupt_partial_starts_clean(tmp_path):
    (tmp_path / partial_mod.PARTIAL_NAME).write_text("{ truncated")
    assert len(_store(tmp_path)) == 0


def test_discard_is_idempotent(tmp_path):
    p = _store(tmp_path)
    p.save()
    assert p.path.exists()
    p.discard()
    p.discard()
    assert not p.path.exists()


def test_fingerprint_is_stable_and_specific():
    a = partial_mod.candidates_fingerprint(CANDS)
    assert a == partial_mod.candidates_fingerprint(list(CANDS))
    assert a != partial_mod.candidates_fingerprint(CANDS[:-1])
    assert a != partial_mod.candidates_fingerprint(
        [{"start": 0.0, "end": 9.0}] + CANDS[1:]
    )


# --- the error that carries the choice ------------------------------------

def test_partial_result_error_carries_counts():
    from publikclip_pipeline.jobs.queue import PartialResultError, StageError

    err = PartialResultError(
        "out of quota", done=19, total=35, stage="score", resume_flag="--partial-ok"
    )
    assert isinstance(err, StageError)  # existing handlers still catch it
    assert err.to_json() == {
        "stage": "score", "done": 19, "total": 35, "resume_flag": "--partial-ok"
    }


def test_settings_round_trip_partial_flag():
    from publikclip_pipeline.config import Settings

    s = Settings()
    assert s.allow_partial_scoring is False
    s.allow_partial_scoring = True
    assert Settings.from_json(json.loads(json.dumps(s.to_json()))).allow_partial_scoring is True
