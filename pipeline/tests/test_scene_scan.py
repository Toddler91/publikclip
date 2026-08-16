"""Scene scan: walked coarsely, and never twice for the same media.

Detecting cuts decodes the whole video, which on a long 60 fps capture costs
more than transcribing it — all to feed the lightest channel in the interest
curve. So it is walked at a fixed rate in seconds, and its result is kept.
"""

from publikclip_pipeline.candidates import stage as candidates


def test_frame_skip_targets_a_fixed_rate_in_seconds():
    assert candidates._frame_skip_for(60.0) == 9  # every 10th frame -> 6 fps
    assert candidates._frame_skip_for(30.0) == 4
    assert candidates._frame_skip_for(24.0) == 3


def test_frame_skip_never_goes_negative_on_slow_sources():
    """A source already at or below the analysis rate is walked whole."""
    assert candidates._frame_skip_for(6.0) == 0
    assert candidates._frame_skip_for(1.0) == 0
    assert candidates._frame_skip_for(0.0) == 0


def test_cache_round_trips(tmp_path):
    key = candidates._scene_cache_key({"source_hash": "abc"})
    assert candidates.load_cached_scenes(tmp_path, key) is None
    candidates.store_cached_scenes(tmp_path, key, [1.0, 2.5])
    assert candidates.load_cached_scenes(tmp_path, key) == [1.0, 2.5]


def test_cache_is_keyed_on_the_media(tmp_path):
    """Never serve one file's cuts for another."""
    candidates.store_cached_scenes(
        tmp_path, candidates._scene_cache_key({"source_hash": "abc"}), [1.0]
    )
    other = candidates._scene_cache_key({"source_hash": "def"})
    assert candidates.load_cached_scenes(tmp_path, other) is None


def test_cache_is_keyed_on_the_analysis_rate(tmp_path, monkeypatch):
    """Walking the same media more finely is a different answer, not a stale
    one — the cache must not hide a changed SCENE_ANALYSIS_FPS."""
    candidates.store_cached_scenes(
        tmp_path, candidates._scene_cache_key({"source_hash": "abc"}), [1.0]
    )
    monkeypatch.setattr(candidates, "SCENE_ANALYSIS_FPS", 12.0)
    finer = candidates._scene_cache_key({"source_hash": "abc"})
    assert candidates.load_cached_scenes(tmp_path, finer) is None


def test_corrupt_cache_is_ignored(tmp_path):
    (tmp_path / candidates.SCENES_CACHE_NAME).write_text("{not json")
    key = candidates._scene_cache_key({"source_hash": "abc"})
    assert candidates.load_cached_scenes(tmp_path, key) is None


def test_missing_cache_dir_does_not_raise(tmp_path):
    """Storing into a job dir that vanished is a slower next run, not a crash."""
    candidates.store_cached_scenes(
        tmp_path / "gone", candidates._scene_cache_key({"source_hash": "abc"}), [1.0]
    )
