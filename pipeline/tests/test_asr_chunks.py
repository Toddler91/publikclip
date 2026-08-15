"""Chunk planning and the partial-transcript store that makes ASR resumable."""

import numpy as np

from publikclip_pipeline.asr import chunks

SR = 16_000


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)


def test_short_audio_is_one_chunk():
    assert chunks.plan(_noise(300), SR) == [(0.0, 300.0)]


def test_spans_are_contiguous_and_total():
    audio = _noise(65 * 60)
    spans = chunks.plan(audio, SR)
    assert len(spans) > 1
    assert spans[0][0] == 0.0
    assert abs(spans[-1][1] - len(audio) / SR) < 1e-6
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert abs(end - start) < 1e-9  # no gaps, no overlap


def test_boundaries_snap_to_silence():
    """A cut through speech loses the word; a cut in a gap is free."""
    audio = _noise(45 * 60)
    silences, target = [], chunks.CHUNK_SEC
    for _ in range(3):
        at = target - 6.0
        audio[int(at * SR):int((at + 0.6) * SR)] = 0.0
        silences.append(at + 0.3)  # centre of the quiet frame
        target = at + 0.3 + chunks.CHUNK_SEC

    cuts = [start for start, _ in chunks.plan(audio, SR)[1:]]
    for cut, silence in zip(cuts, silences):
        # accurate to the frame length we scan at
        assert abs(cut - silence) <= chunks.FRAME_SEC / 2 + 1e-6


def test_no_chunk_below_minimum():
    for seconds in (chunks.CHUNK_SEC + 1, chunks.CHUNK_SEC * 2.5, 4000):
        spans = chunks.plan(_noise(seconds), SR)
        assert all(end - start >= chunks.MIN_CHUNK_SEC for start, end in spans), seconds


def test_empty_audio_does_not_crash():
    assert chunks.plan(np.zeros(0, dtype=np.float32), SR) == [(0.0, 0.0)]


# --- partial store -------------------------------------------------------

SPANS = [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1500.0)]


def _partial(tmp_path, fingerprint="fp", spans=None):
    return chunks.Partial.load_or_new(
        tmp_path / chunks.PARTIAL_NAME, fingerprint, spans or SPANS
    )


def test_partial_round_trips(tmp_path):
    p = _partial(tmp_path)
    p.language = "en"
    p.transcribed[0] = [{"start": 1.0, "end": 2.0, "text": "hello"}]
    p.save()

    again = _partial(tmp_path)
    assert again.language == "en"
    assert again.transcribed[0][0]["text"] == "hello"
    assert again.transcribe_done() == 1
    assert again.n_chunks == 3


def test_merge_offsets_to_absolute_time(tmp_path):
    p = _partial(tmp_path)
    p.aligned[1] = [{
        "start": 5.0, "end": 6.0, "text": "world",
        "words": [{"word": "world", "start": 5.0, "end": 6.0, "score": 0.8}],
    }]
    merged = p.merged_segments()
    assert merged[0]["start"] == 605.0          # chunk 1 starts at 600 s
    assert merged[0]["words"][0]["start"] == 605.0


def test_merge_sorts_and_skips_wordless(tmp_path):
    p = _partial(tmp_path)
    p.aligned[2] = [{"start": 1.0, "end": 2.0, "text": "late", "words": []}]
    p.aligned[0] = [{"start": 9.0, "end": 9.5, "text": "early", "words": []}]
    starts = [s["start"] for s in p.merged_segments()]
    assert starts == sorted(starts)


def test_different_audio_discards_partial(tmp_path):
    p = _partial(tmp_path)
    p.transcribed[0] = [{"start": 0.0, "end": 1.0, "text": "x"}]
    p.save()
    assert _partial(tmp_path, fingerprint="other").transcribe_done() == 0


def test_changed_plan_discards_partial(tmp_path):
    p = _partial(tmp_path)
    p.transcribed[0] = [{"start": 0.0, "end": 1.0, "text": "x"}]
    p.save()
    moved = [(0.0, 300.0), (300.0, 1200.0), (1200.0, 1500.0)]
    assert _partial(tmp_path, spans=moved).transcribe_done() == 0


def test_corrupt_partial_starts_clean(tmp_path):
    (tmp_path / chunks.PARTIAL_NAME).write_text("{ truncated")
    assert _partial(tmp_path).transcribe_done() == 0


def test_discard_removes_file(tmp_path):
    p = _partial(tmp_path)
    p.save()
    assert p.path.exists()
    p.discard()
    assert not p.path.exists()
    p.discard()  # idempotent
