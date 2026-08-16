"""Gameplay audio scored against its own baseline, not by presence.

In a shooter, gunfire is the background. A detector that scores it raw marks
the whole VOD equally interesting, which is the same as marking none of it.
"""

import numpy as np

from publikclip_pipeline.candidates import curve


def _fire(start: float, end: float, conf: float = 1.0) -> dict:
    return {"type": "gunfire", "start": start, "end": end, "confidence": conf}


def test_constant_firefight_is_not_a_highlight():
    """Forty minutes of sustained fire is forty minutes of normal."""
    n = 900
    timeline = [_fire(t, t + 1) for t in range(n)]
    out = curve.gameplay_events_channel(timeline, n)
    assert float(out.max()) == 0.0


def test_a_burst_out_of_quiet_spikes():
    n = 900
    timeline = [_fire(t, t + 1) for t in range(400, 430)]
    out = curve.gameplay_events_channel(timeline, n)
    assert float(out.max()) > 0.5
    assert 380 <= int(np.argmax(out)) <= 450


def test_the_quiet_stretch_scores_below_the_burst():
    n = 900
    timeline = [_fire(t, t + 1) for t in range(400, 430)]
    out = curve.gameplay_events_channel(timeline, n)
    assert float(out[:300].max()) < float(out[400:430].max())


def test_busier_than_usual_beats_steady_fire():
    """A heavy exchange inside constant fire still reads as a moment."""
    n = 900
    timeline = [_fire(t, t + 1, conf=0.3) for t in range(n)]
    timeline += [_fire(t, t + 1, conf=1.0) for t in range(500, 520)]
    out = curve.gameplay_events_channel(timeline, n)
    assert float(out[480:540].max()) > float(out[:300].max())


def test_empty_timeline_is_silent():
    assert float(curve.gameplay_events_channel([], 600).max()) == 0.0
    assert len(curve.gameplay_events_channel([], 600)) == 600


def test_zero_length_video_does_not_raise():
    assert len(curve.gameplay_events_channel([_fire(0, 1)], 0)) == 0


def test_conversational_events_are_not_counted_here():
    """Laughter belongs to the events channel; counting it twice would weigh
    one detection on two different scales."""
    n = 600
    timeline = [{"type": "laugh", "start": float(t), "end": t + 1.0} for t in range(100, 130)]
    assert float(curve.gameplay_events_channel(timeline, n).max()) == 0.0


def test_gunfire_is_absent_from_the_conversational_event_weights():
    for name in curve.GAMEPLAY_EVENT_WEIGHTS:
        assert name not in curve.EVENT_WEIGHTS


def test_weights_target_solo_gameplay():
    """The audience-dependent channels are pinned off, not merely small."""
    assert curve.WEIGHTS["heatmap"] == 0.0   # no chat
    assert curve.WEIGHTS["turns"] == 0.0     # one speaker
    assert curve.WEIGHTS["gameplay_events"] > 0.0
    # Vocal energy carries the most weight of any live channel.
    live = {k: v for k, v in curve.WEIGHTS.items() if v > 0}
    assert max(live, key=live.get) == "dynamics"


def test_every_weighted_channel_is_one_the_stage_builds():
    """A weight for a channel nobody produces is silently dead."""
    from publikclip_pipeline.candidates import stage  # noqa: F401

    built = {
        "heatmap", "dynamics", "events", "turns",
        "arousal", "scenes", "lexical", "gameplay_events",
    }
    assert set(curve.WEIGHTS) == built
