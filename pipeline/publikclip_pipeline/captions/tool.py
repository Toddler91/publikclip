"""Captions only: a video in, the same video with burned-in captions out.

The clipper's caption look — word-by-word highlighting, prosodic emphasis,
[laughs] tags — without any of the clipping. No candidate scoring, no LLM, no
camera direction, no cutting: the source comes back whole, captioned.

It runs on the pipeline's own ingest and ASR stages rather than a private
transcription path, which buys chunk-level checkpointing on a long source and
makes a second run against the same file nearly free. That matters here more
than it looks: transcription is the entire cost of this tool.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .. import config
from ..events import dsp
from ..jobs import queue
from ..render import ffmpeg_bin, renderer
from . import ass as ass_mod

DEFAULT_SUFFIX = ".captioned.mp4"


class CaptionError(Exception):
    """Something the user can act on: no ffmpeg, no speech, unreadable source."""


def _find_existing_job(source: Path) -> queue.Job | None:
    """The most recent job for this exact file, so its transcript is reused.

    Keyed on the resolved path: the same video captioned twice should cost one
    transcription, and re-running after a crash should cost none.
    """
    target = str(source.resolve())
    for job in queue.list_jobs(limit=200):
        # Caption jobs only. A clip job for the same file has a transcript we
        # could reuse, but adopting its directory would retitle it and mark it
        # done as though the clipping run had finished.
        if job.kind == "caption" and job.source_type == "file" and job.source == target:
            return job
    return None


def _read_analysis_audio(path: Path):
    """The 16 kHz mono analysis WAV, as float32 in [-1, 1).

    Deliberately not librosa.load. librosa lazily imports `samplerate`, whose
    loader walks the call stack with inspect; that touches speechbrain's lazy
    module proxy, left loaded by alignment, which tries to import k2 — not a
    dependency here — and the read dies several libraries from anything to do
    with audio. Ingest writes this file itself as pcm_s16le mono, so the
    stdlib reads it with no import surface at all.
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise CaptionError(f"Expected 16-bit PCM analysis audio, got {path}")
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def _words_for(segments: list[dict]) -> list[ass_mod.Word]:
    """Whole-source word list. Times stay absolute — there is no clip to be
    relative to, which is the one place this differs from the render stage."""
    words: list[ass_mod.Word] = []
    for seg in segments:
        for w in seg.get("words", []):
            if w.get("start") is None or w.get("end") is None:
                continue  # alignment drops a word it could not place
            words.append(
                ass_mod.Word(text=w["word"], start=float(w["start"]), end=float(w["end"]))
            )
    return words


def _burn(source: Path, ass_path: Path, out_path: Path, timeout: float) -> None:
    args = [
        ffmpeg_bin.ffmpeg(), "-y", "-v", "error", "-i", str(source),
        # Explicit: these sources carry up to six OBS audio tracks, and an
        # unqualified selection silently picks one of them.
        "-map", "0:v:0", "-map", "0:a:0",
        "-vf", f"subtitles=filename={renderer._q(ass_path)}"  # noqa: SLF001
               f":fontsdir={renderer._q(ass_mod.FONTS_DIR)}",  # noqa: SLF001
        "-c:v", "libx264", "-preset", "medium", "-crf", str(renderer.X264_CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise CaptionError(f"Burn-in failed: {(proc.stderr or '')[-800:]}")


def caption_video(
    source: Path,
    out_path: Path | None = None,
    preset: str = "classic",
    tags: bool = False,
    ass_only: bool = False,
    progress=None,
    timeout: float = 7200.0,
) -> dict:
    """Transcribe `source` and return it captioned. Returns a small summary."""
    from ..asr.stage import AsrStage
    from ..ingest.stage import IngestStage

    emit = progress or (lambda stage, fraction, message: None)

    source = Path(source)
    if not source.exists():
        raise CaptionError(f"No such file: {source}")
    if preset not in ass_mod.PRESETS:
        raise CaptionError(
            f"Unknown preset {preset!r}. Available: {', '.join(sorted(ass_mod.PRESETS))}"
        )
    if ffmpeg_bin.ensure_present(lambda f, m: emit("ffmpeg", f, m)) is None:
        raise CaptionError(
            "No usable ffmpeg: none on PATH, none in PUBLIKCLIP_HOME/bin, and "
            "fetching a static build failed."
        )

    job = _find_existing_job(source)
    if job is None:
        settings_json = json.dumps(config.Settings().to_json())
        job = queue.create_job("file", str(source.resolve()), settings_json, kind="caption")

    stages = [IngestStage(), AsrStage()]
    if tags:
        from ..events.stage import EventsStage

        stages.append(EventsStage())
    results = queue.run_stages(job, stages, emit)

    ingest, asr = results["ingest"], results["asr"]
    words = _words_for(asr["segments"])
    if not words:
        raise CaptionError("No speech found — nothing to caption.")

    emit("captions", 0.9, "Building captions…")
    curves = dsp.energy_curves(_read_analysis_audio(Path(ingest["audio_path"])))
    ass_mod.mark_emphasis(words, curves["rms"], float(curves["grid_sec"]), clip_start=0.0)

    events = []
    if tags:
        events = [
            {"type": e["type"], "start": e["start"], "end": e["end"]}
            for e in results["events"]["timeline"]
            if e["type"] != "pause"
        ]

    probe = ingest["probe"]
    document = ass_mod.build_ass(
        words, events,
        preset_name=preset,
        emoji_ok=ass_mod.emoji_probe(),
        play_res=(int(probe["width"]), int(probe["height"])),
    )

    out_path = Path(out_path) if out_path else source.with_suffix(DEFAULT_SUFFIX)
    ass_path = out_path.with_suffix(".ass")
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(document, encoding="utf-8")

    summary = {
        "job_id": job.id,
        "source": str(source),
        "ass_path": str(ass_path),
        "words": len(words),
        "event_tags": len(events),
        "preset": preset,
        "play_res": [int(probe["width"]), int(probe["height"])],
        "burned": False,
        "output": None,
    }
    if ass_only:
        queue.set_job_status(job.id, "done", title=source.stem)
        return summary

    if not ffmpeg_bin.supports_captions():
        raise CaptionError(
            "This ffmpeg has no subtitles filter, so captions cannot be burned in. "
            f"The subtitle file is written: {ass_path}"
        )
    emit("captions", 0.95, "Burning captions in…")
    _burn(source, ass_path, out_path, timeout)
    # Once burned in, the subtitle file is an intermediate. Restyling is nearly
    # free — the transcript is cached against the job — so keeping it would
    # only litter the output folder. --ass-only is how you ask to keep it.
    ass_path.unlink(missing_ok=True)
    summary["ass_path"] = None
    summary["burned"] = True
    summary["output"] = str(out_path)
    # Otherwise the row keeps whatever the stage machinery last wrote, and the
    # app lists a finished caption as a failed session forever.
    queue.set_job_status(job.id, "done", title=source.stem)
    return summary
