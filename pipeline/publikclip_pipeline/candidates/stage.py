"""Candidates stage: build every free channel, weigh them into the interest
curve, extract ~35 sentence-snapped candidate windows. No LLM spend here —
this count is the cost gate for T1/T2."""

from __future__ import annotations

import json
from pathlib import Path

from ..jobs.queue import Stage, StageContext, StageError


SCENES_CACHE_NAME = "scenes.cache.json"

# Frames per second the detector actually examines. The scenes channel is
# weighted 0.05 and smoothed over a 20 s window, so a cut located to a sixth
# of a second is already far finer than anything downstream can use.
SCENE_ANALYSIS_FPS = 6.0


def _frame_skip_for(frame_rate: float) -> int:
    """Frames to step over so the detector sees ~SCENE_ANALYSIS_FPS per second.

    Derived from the source rate rather than fixed: a 60 fps stream capture
    and a 24 fps upload should be walked at the same rate in seconds, not the
    same rate in frames.
    """
    if not frame_rate or frame_rate <= SCENE_ANALYSIS_FPS:
        return 0
    return max(0, int(round(frame_rate / SCENE_ANALYSIS_FPS)) - 1)


def detect_scenes(media_path: str, progress=None) -> list[float]:
    """Scene-change timestamps via PySceneDetect ContentDetector (BSD-3) on
    a downscaled decode. On a static podcast this returns camera cuts; on
    gaming/vlog footage it captures visual pacing.

    auto_downscale shrinks the image the content metric runs on, but without a
    frame skip every frame is still fully decoded and converted — a quarter of
    a million of them on an hour of 60 fps capture, to feed the cheapest
    channel in the curve. Step over most of them instead.
    """
    from scenedetect import ContentDetector, open_video
    from scenedetect.scene_manager import SceneManager

    video = open_video(media_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=27.0))
    manager.auto_downscale = True
    manager.detect_scenes(
        video, show_progress=False, frame_skip=_frame_skip_for(video.frame_rate)
    )
    return [start.get_seconds() for start, _ in manager.get_scene_list()]


def _scene_cache_key(ingest: dict) -> str:
    """What the scan's answer depends on: which media, walked how finely."""
    media = ingest.get("source_hash") or ingest.get("media_path", "")
    return f"{media}@{SCENE_ANALYSIS_FPS}"


def load_cached_scenes(job_dir: Path, key: str) -> list[float] | None:
    """A previous scan of this same media at this same rate, or None.

    Scanning is by far the most expensive thing this stage does, and nothing
    about it changes between attempts. Without this, a stage that failed after
    the scan — or a job resumed after the app restarted — pays for the whole
    decode again, which on a long capture costs more than transcription did.
    """
    try:
        blob = json.loads((job_dir / SCENES_CACHE_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if blob.get("key") != key:
        return None
    times = blob.get("times")
    return times if isinstance(times, list) else None


def store_cached_scenes(job_dir: Path, key: str, times: list[float]) -> None:
    try:
        (job_dir / SCENES_CACHE_NAME).write_text(json.dumps({"key": key, "times": times}))
    except OSError:
        pass  # a cache we could not write is a slower next run, not a failure


class CandidatesStage(Stage):
    name = "candidates"
    schema_version = 1

    def run(self, ctx: StageContext) -> dict:
        import numpy as np

        from . import curve as curve_mod
        from . import windows as windows_mod

        prior = ctx.prior or {}
        ingest = prior.get("ingest")
        diarize = prior.get("diarize")
        events = prior.get("events")
        if not (ingest and diarize and events):
            raise StageError("Candidates need ingest + diarize + events outputs.")

        segments = diarize["segments"]
        duration = float(ingest["probe"]["duration_sec"])
        n = int(np.ceil(duration))

        curves_path = Path(events["curves_path"])
        if not curves_path.exists():
            raise StageError("curves.json missing — re-run events.")
        curves = json.loads(curves_path.read_text())

        cache_key = _scene_cache_key(ingest)
        scene_times = load_cached_scenes(ctx.job_dir, cache_key)
        if scene_times is None:
            ctx.emit(-1, "Detecting scene changes…")
            try:
                scene_times = detect_scenes(ingest["media_path"])
            except Exception:  # noqa: BLE001 — scenes are a minor channel; degrade
                scene_times = []
            else:
                # Only a completed scan is worth keeping. A failure here is
                # usually transient, and caching [] would silence the channel
                # for the rest of the job's life.
                store_cached_scenes(ctx.job_dir, cache_key, scene_times)
        else:
            ctx.emit(0.5, f"Scene changes cached ({len(scene_times)} cuts)")
        (ctx.job_dir / "scenes.json").write_text(json.dumps(scene_times))

        ctx.emit(0.6, "Building interest curve…")
        channels = {
            "heatmap": curve_mod.heatmap_channel(ingest.get("heatmap"), n),
            "dynamics": curve_mod.dynamics_channel(curves["dynamics"], curves["grid_sec"], n),
            "events": curve_mod.events_channel(events["timeline"], n),
            "turns": curve_mod.turns_channel(diarize["turns"], n),
            "arousal": curve_mod.arousal_channel(
                curves.get("arousal", []), curves.get("arousal_grid_sec", 0.5), n
            ),
            "scenes": curve_mod.scenes_channel(scene_times, n),
            "lexical": curve_mod.lexical_channel(segments, n),
            "gameplay_events": curve_mod.gameplay_events_channel(events["timeline"], n),
        }
        curve, effective_weights = curve_mod.interest_curve(channels)

        ctx.emit(0.8, "Extracting candidate windows…")
        candidates = windows_mod.extract(curve, channels, segments, duration)
        if not candidates:
            raise StageError(
                "No candidate moments found — the video may be too short or too quiet."
            )

        # Persist the curve for the review UI's timeline visualization.
        (ctx.job_dir / "interest_curve.json").write_text(
            json.dumps({"per_sec": np.round(curve, 4).tolist()})
        )

        return {
            "candidates": [c.to_json() for c in candidates],
            "count": len(candidates),
            "effective_weights": effective_weights,
            "scene_count": len(scene_times),
            "heatmap_present": bool(ingest.get("heatmap")),
        }
