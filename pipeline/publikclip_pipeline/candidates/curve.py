"""The free interest curve — the cost gate for everything downstream.

Weighted sum of local signals on a 1 s grid, weights anchored to the SHAP
importances reported in arXiv 2512.21402 (see RESEARCH.md): audio energy
dynamics and engagement (heatmap) dominate, event density and speaker-turn
rate follow, arousal and lexical power-words trail.

License note: the research draft named the NRC-VAD lexicon for the lexical
channel, but NRC lexicons are research-only (commercial use requires a
paid license) — incompatible with AGPL redistribution. Swapped for a small
built-in power-word list; recorded as a deliberate substitution.
"""

from __future__ import annotations

import numpy as np

# Channel weights, normalized in interest_curve() — ratios, not fractions that
# must sum to 1.
#
# Retuned for solo gameplay with no live audience. The SHAP-anchored table this
# replaces (arXiv 2512.21402) was fitted on conversational video with a crowd,
# and two of its strongest channels are structurally dead here rather than
# merely weak: heatmap needs chat that does not exist, and turns needs a second
# speaker. interest_curve() would redistribute both away on its own once they
# came back all-zero — they are pinned at 0 to say so deliberately rather than
# by accident, and their code paths stay wired for the day there is a crowd.
#
# What remains: the streamer's own voice carries most of the signal, and
# gameplay audio carries the part that does not depend on them reacting to it.
WEIGHTS = {
    "dynamics": 0.34,          # vocal energy — the primary signal now
    "gameplay_events": 0.22,   # gunfire/explosions, against their own baseline
    "scenes": 0.16,            # deaths, killcams, respawns
    "events": 0.14,            # whoops, laughs, shouts
    "arousal": 0.11,
    "lexical": 0.03,           # only nudges
    "heatmap": 0.00,           # no chat
    "turns": 0.00,             # one speaker
}

# Compact power-word list (built-in, license-clean). Lexical is the weakest
# channel by design — it only nudges.
POWER_WORDS = {
    "insane", "crazy", "unbelievable", "never", "secret", "nobody", "worst",
    "best", "free", "money", "dead", "died", "killed", "scared", "terrified",
    "shocked", "shocking", "literally", "actually", "truth", "lie", "lied",
    "caught", "exposed", "banned", "illegal", "million", "billion", "broke",
    "rich", "hate", "hated", "love", "loved", "wild", "weird", "crime",
    "arrested", "fight", "fought", "cried", "screamed", "blew", "exploded",
}

EVENT_WEIGHTS = {"laugh": 1.0, "gasp": 0.9, "scream": 0.8, "cheer": 0.7, "applause": 0.6, "shout": 0.5}

# Gameplay audio is deliberately absent from EVENT_WEIGHTS above: events_channel
# skips types it does not know, so gunfire flows to its own channel instead of
# being counted twice on two different scales.
GAMEPLAY_EVENT_WEIGHTS = {"gunfire": 1.0, "explosion": 1.0}

# Seconds of context defining "normal" for the gameplay channel. Long enough
# that a single firefight cannot drag the baseline up around itself.
GAMEPLAY_BASELINE_SEC = 121


def _norm(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    top = np.percentile(x, 98)
    if top <= 0:
        return np.zeros_like(x)
    return np.clip(x / top, 0.0, 1.0)


def heatmap_channel(heatmap: list[dict] | None, n: int) -> np.ndarray:
    out = np.zeros(n)
    if not heatmap:
        return out
    for seg in heatmap:
        a, b = int(seg["start_time"]), int(np.ceil(seg["end_time"]))
        out[max(0, a) : min(n, b)] = np.maximum(out[max(0, a) : min(n, b)], seg["value"])
    return out


def dynamics_channel(dynamics: list[float], grid_sec: float, n: int) -> np.ndarray:
    arr = np.asarray(dynamics, dtype=float)
    if len(arr) == 0:
        return np.zeros(n)
    per_sec = max(1, int(round(1.0 / grid_sec)))
    trimmed = arr[: (len(arr) // per_sec) * per_sec]
    coarse = trimmed.reshape(-1, per_sec).mean(axis=1)
    out = np.zeros(n)
    out[: min(n, len(coarse))] = coarse[:n]
    return _norm(out)


def events_channel(timeline: list[dict], n: int) -> np.ndarray:
    """Weighted event presence, with a ±2 s halo — the moment AROUND a laugh
    is what clips well, not just the laugh itself."""
    out = np.zeros(n)
    for event in timeline:
        w = EVENT_WEIGHTS.get(event["type"])
        if w is None:
            continue
        w *= float(event.get("confidence", 1.0))
        a = max(0, int(event["start"]) - 2)
        b = min(n, int(np.ceil(event["end"])) + 2)
        out[a:b] = np.maximum(out[a:b], w)
    return out


def gameplay_events_channel(
    timeline: list[dict],
    n: int,
    window: int = 10,
    baseline_sec: int = GAMEPLAY_BASELINE_SEC,
) -> np.ndarray:
    """Gameplay audio density measured against its own rolling baseline.

    Presence is the wrong question. In a shooter, gunfire IS the background:
    scored raw it would mark the entire VOD equally interesting, which is the
    same as marking none of it. What separates a moment is firing *unlike the
    surrounding minutes* — a burst out of quiet, or a sudden heavy exchange in
    a lull. A player camping a quiet corner for ten minutes and then winning a
    fight should spike; a player in a constant 40-minute firefight should not
    read as 40 minutes of highlight.

    Median and MAD rather than mean and standard deviation, because the thing
    being looked for is precisely the outlier — it would inflate both of those
    and so partly hide itself.
    """
    from scipy.ndimage import median_filter

    if n <= 0:
        return np.zeros(max(0, n))

    density = np.zeros(n)
    for event in timeline:
        w = GAMEPLAY_EVENT_WEIGHTS.get(event["type"])
        if w is None:
            continue
        w *= float(event.get("confidence", 1.0))
        a = max(0, int(event["start"]))
        b = min(n, int(np.ceil(event["end"])) + 1)
        if a < b:
            density[a:b] += w
    if not density.any():
        return np.zeros(n)

    density = np.convolve(density, np.ones(window) / window, mode="same")
    size = max(1, min(n, baseline_sec if baseline_sec % 2 else baseline_sec + 1))
    baseline = median_filter(density, size=size, mode="nearest")
    spread = median_filter(np.abs(density - baseline), size=size, mode="nearest")
    # Stretches of perfectly flat audio have zero spread, which would divide
    # every deviation into infinity. Fall back to the typical spread elsewhere
    # in the video so a quiet-then-loud VOD still resolves sensibly.
    nonzero = spread[spread > 0]
    floor = float(np.median(nonzero)) if nonzero.size else 1e-6
    return _norm(np.clip((density - baseline) / np.maximum(spread, floor), 0.0, None))


def turns_channel(turns: list[dict], n: int, window: int = 15) -> np.ndarray:
    """Speaker-change rate in a sliding window."""
    changes = np.zeros(n)
    for prev, cur in zip(turns, turns[1:]):
        if prev["speaker"] != cur["speaker"]:
            t = int(cur["start"])
            if 0 <= t < n:
                changes[t] += 1.0
    kernel = np.ones(window)
    density = np.convolve(changes, kernel, mode="same")
    return _norm(density)


def arousal_channel(arousal: list[float], grid_sec: float, n: int) -> np.ndarray:
    arr = np.asarray(arousal, dtype=float)
    if len(arr) == 0:
        return np.zeros(n)
    per_sec = max(1, int(round(1.0 / grid_sec)))
    if per_sec > 1:
        trimmed = arr[: (len(arr) // per_sec) * per_sec]
        coarse = trimmed.reshape(-1, per_sec).mean(axis=1)
    else:
        # grid coarser than 1 s → repeat
        coarse = np.repeat(arr, int(round(grid_sec)))
    out = np.zeros(n)
    out[: min(n, len(coarse))] = coarse[:n]
    return _norm(out)


def scenes_channel(scene_times: list[float], n: int, window: int = 20) -> np.ndarray:
    marks = np.zeros(n)
    for t in scene_times:
        if 0 <= int(t) < n:
            marks[int(t)] += 1.0
    density = np.convolve(marks, np.ones(window), mode="same")
    return _norm(density)


def lexical_channel(segments: list[dict], n: int) -> np.ndarray:
    out = np.zeros(n)
    for seg in segments:
        for word in seg.get("words", []):
            token = word["word"].lower().strip(".,!?\"'")
            if token in POWER_WORDS:
                t = int(word["start"])
                if 0 <= t < n:
                    out[max(0, t - 1) : min(n, t + 2)] += 0.5
    return _norm(out)


def interest_curve(channels: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    """Weighted sum. Missing channels (all-zero heatmap on most videos) get
    their weight redistributed proportionally so the curve never silently
    deflates — and the redistribution is reported for provenance."""
    n = max(len(c) for c in channels.values())
    active = {k: v for k, v in channels.items() if len(v) and float(np.max(v)) > 0}
    total_w = sum(WEIGHTS[k] for k in active)
    if total_w == 0:
        return np.zeros(n), {}
    effective = {k: WEIGHTS[k] / total_w for k in active}
    curve = np.zeros(n)
    for name, weight in effective.items():
        ch = channels[name]
        padded = np.zeros(n)
        padded[: len(ch)] = ch
        curve += weight * padded
    return curve, {k: round(v, 4) for k, v in effective.items()}
