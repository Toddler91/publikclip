"""publikclip CLI.

Doubles as the desktop app's sidecar: with --jsonl every progress event and
the final result are emitted as one JSON object per stdout line, so the
Tauri shell just spawns `publikclip --jsonl run <source>` and streams.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .jobs import queue, runlock, supervise


def _stages() -> list[queue.Stage]:
    # Grows per milestone: ingest → asr → diarize → events → candidates →
    # score → camera → render. Stage imports are deferred so `publikclip
    # jobs` doesn't pay the torch import tax.
    from .asr.stage import AsrStage
    from .camera.stage import CameraStage
    from .candidates.stage import CandidatesStage
    from .diarize.stage import DiarizeStage
    from .events.stage import EventsStage
    from .ingest.stage import IngestStage
    from .render.stage import RenderStage
    from .scoring.stage import ScoreStage

    return [
        IngestStage(),
        AsrStage(),
        DiarizeStage(),
        EventsStage(),
        CandidatesStage(),
        ScoreStage(),
        CameraStage(),
        RenderStage(),
    ]


def _progress_printer(jsonl: bool):
    def emit(stage: str, fraction: float, message: str) -> None:
        if jsonl:
            print(
                json.dumps(
                    {"event": "progress", "stage": stage, "fraction": fraction, "message": message}
                ),
                flush=True,
            )
        else:
            pct = f"{fraction * 100:5.1f}%" if fraction >= 0 else "  ...."
            print(f"[{stage:<10}] {pct} {message}", file=sys.stderr, flush=True)

    return emit


def _emit_result(jsonl: bool, payload: dict) -> None:
    if jsonl:
        print(json.dumps({"event": "result", **payload}), flush=True)
    else:
        print(json.dumps(payload, indent=2))


def cmd_run(args: argparse.Namespace) -> int:
    source = args.source
    source_type = "url" if source.startswith(("http://", "https://")) else "file"
    settings = config.Settings()
    if args.llm:
        settings.llm_mode = args.llm
    if args.captions:
        settings.caption_preset = args.captions
    if args.camera:
        settings.camera.speaker_change = args.camera
    settings.allow_partial_scoring = bool(getattr(args, "partial_ok", False))
    job = queue.create_job(source_type, source, json.dumps(settings.to_json()))
    return _execute(job, args.jsonl)


def cmd_caption(args: argparse.Namespace) -> int:
    """Captions only — the source back whole, with captions burned in."""
    from .captions import ass as ass_mod
    from .captions import tool as caption_tool

    if args.list_presets:
        for name in sorted(ass_mod.PRESETS):
            print(name)
        return 0
    if not args.source:
        print("caption: a source file is required", file=sys.stderr)
        return 2

    if args.jsonl:
        # Sidecar mode: exit if the app reading our stdout goes away, rather
        # than re-encoding an hour of video for nobody.
        supervise.die_with_parent()

    def announce(job: queue.Job) -> None:
        if args.jsonl:
            print(json.dumps({
                "event": "job", "job_id": job.id, "dir": str(job.dir), "kind": "caption",
            }), flush=True)

    try:
        summary = caption_tool.caption_video(
            Path(args.source),
            out_path=Path(args.output) if args.output else None,
            preset=args.preset,
            tags=args.tags,
            ass_only=args.ass_only,
            progress=_progress_printer(args.jsonl),
            on_job=announce,
        )
    except (caption_tool.CaptionError, queue.StageError, runlock.JobBusyError) as exc:
        # The shell reads `ok` to tell a finished run from a failed one; without
        # it every result, including a good one, reads as a failure.
        _emit_result(args.jsonl, {"ok": False, "kind": "caption", "error": str(exc)})
        if not args.jsonl:
            print(f"caption: {exc}", file=sys.stderr)
        return 1
    _emit_result(args.jsonl, {"ok": True, "kind": "caption", **summary})
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    job = queue.get_job(args.job_id)
    if job is None:
        print(f"No job {args.job_id}", file=sys.stderr)
        return 2
    partial_ok = bool(getattr(args, "partial_ok", False))
    if args.llm or args.captions or args.camera or partial_ok:
        settings = config.Settings.from_json(json.loads(job.settings_json))
        if args.llm:
            settings.llm_mode = args.llm
        if args.captions:
            settings.caption_preset = args.captions
        if args.camera:
            settings.camera.speaker_change = args.camera
        # Sticky only when asked for: a later plain resume goes back to
        # demanding a complete scoring pass.
        settings.allow_partial_scoring = partial_ok
        new_json = json.dumps(settings.to_json())
        with queue._connect() as conn:  # noqa: SLF001 — CLI is a queue friend
            conn.execute("UPDATE jobs SET settings_json = ? WHERE id = ?", (new_json, job.id))
        job = queue.get_job(args.job_id)
    return _execute(job, args.jsonl)


def _execute(job: queue.Job, jsonl: bool) -> int:
    emit = _progress_printer(jsonl)
    if jsonl:
        # Sidecar mode: exit if the app reading our stdout goes away, rather
        # than transcribing on for nobody and colliding with the next run.
        supervise.die_with_parent()
        print(json.dumps({"event": "job", "job_id": job.id, "dir": str(job.dir)}), flush=True)
    else:
        print(f"job {job.id} → {job.dir}", file=sys.stderr)
    try:
        results = queue.run_stages(job, _stages(), emit)
    except runlock.JobBusyError as err:
        _emit_result(jsonl, {"ok": False, "job_id": job.id, "error": str(err), "busy": True})
        return 4
    except queue.PartialResultError as err:
        _emit_result(jsonl, {
            "ok": False, "job_id": job.id, "error": str(err), "partial": err.to_json(),
        })
        return 1
    except queue.StageError as err:
        _emit_result(jsonl, {"ok": False, "job_id": job.id, "error": str(err)})
        return 1
    summary = {
        "ok": True,
        "job_id": job.id,
        "stages": list(results.keys()),
        "title": results.get("ingest", {}).get("title"),
        "heatmap_segments": len(results.get("ingest", {}).get("heatmap") or []),
    }
    _emit_result(jsonl, summary)
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Forget jobs: their rows, and their directories unless --keep-files."""
    results, failed = [], 0
    for job_id in args.job_ids:
        try:
            results.append(queue.delete_job(job_id, remove_files=not args.keep_files))
        except KeyError:
            print(f"No job {job_id}", file=sys.stderr)
            failed += 1
        except runlock.JobBusyError as exc:
            print(str(exc), file=sys.stderr)
            failed += 1
    freed = sum(r["freed_bytes"] for r in results)
    _emit_result(args.jsonl, {
        "deleted": [r["id"] for r in results],
        "freed_mb": round(freed / (1024 * 1024), 1),
        "failed": failed,
    })
    return 1 if failed else 0


def cmd_jobs(args: argparse.Namespace) -> int:
    for job in queue.list_jobs():
        stages = queue.stage_statuses(job.id)
        done = sum(1 for s in stages.values() if s == "done")
        print(f"{job.id}  {job.status:<8} {done} stage(s) done  {job.title or job.source}")
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    """Every job with its *real* state — what the process table says, not what
    the DB last managed to record. JSON for the app, a table for humans."""
    rows = []
    for job in queue.list_jobs():
        lock = runlock.owner(job.dir)
        stages = queue.stage_statuses(job.id)
        rows.append({
            "job_id": job.id,
            "title": job.title,
            "kind": job.kind,
            "state": queue.job_state(job.id),
            "stage": (lock.stage if lock else None)
            or next((s for s, st in stages.items() if st == "running"), None),
            "pid": lock.pid if lock else None,
            "started_at": lock.started_at if lock else None,
            "stages": stages,
            "error": job.error,
        })
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no jobs yet")
        return 0
    for r in rows:
        pid = f"pid {r['pid']}" if r["pid"] else "-"
        stage = r["stage"] or "-"
        print(f"{r['state']:<12} {stage:<11} {pid:<10} {r['job_id']}  {r['title'] or ''}")
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    """Per-clip editing verbs. All output is JSON on stdout for the app."""
    from pathlib import Path

    from .edits import render_clip as rc
    from .edits import store, visuals

    job = queue.get_job(args.job_id)
    if job is None:
        print(json.dumps({"ok": False, "error": f"no job {args.job_id}"}))
        return 2
    job_dir = Path(job.dir)

    if args.edit_cmd == "context":
        print(json.dumps({"ok": True, **rc.context_for_clip(job_dir, args.clip)}))
        return 0

    if args.edit_cmd == "suggest-visuals":
        score = json.loads((job_dir / "score.json").read_text())["data"]
        clip = score["clips"][args.clip]
        edit = store.edit_for_clip(job_dir, args.clip, clip)
        # plan against OUTPUT-time words = current bounds without dead-space
        # (suggestions land on the source-bounds timeline the UI shows)
        diarize = json.loads((job_dir / "diarize.json").read_text())["data"]
        words = [
            {"word": w["word"], "start": w["start"] - edit.start, "end": w["end"] - edit.start}
            for seg in diarize["segments"]
            for w in seg.get("words", [])
            if edit.start <= w["start"] < edit.end
        ]
        settings = config.Settings.from_json(json.loads(job.settings_json))
        try:
            suggestions = visuals.suggest(job_dir, words, settings.llm_mode, prefer=args.prefer)
        except Exception as err:  # noqa: BLE001 — surface, don't crash the app
            print(json.dumps({"ok": False, "error": str(err)}))
            return 1
        edits = store.load(job_dir)
        current = edits.get(str(args.clip), edit)
        known = {o.id for o in current.overlays}
        current.overlays.extend(o for o in suggestions if o.id not in known)
        edits[str(args.clip)] = current
        store.save(job_dir, edits)
        print(json.dumps({"ok": True, "edit": current.to_json()}))
        return 0

    if args.edit_cmd == "render-clip":
        emit = _progress_printer(args.jsonl)
        try:
            entry = rc.render_clip_edit(job_dir, args.clip, lambda f, m: emit("render", f, m))
        except Exception as err:  # noqa: BLE001
            _emit_result(args.jsonl, {"ok": False, "error": str(err)})
            return 1
        _emit_result(args.jsonl, {"ok": True, "output": entry})
        return 0
    return 2


def cmd_ig(args: argparse.Namespace) -> int:
    from .insights import calibration, instagram

    if args.ig_cmd == "connect":
        conn = instagram.connect(args.app_id, args.app_secret)
        print(f"Connected as @{conn['username']} (user {conn['user_id']}).")
        return 0

    # App-facing commands: exactly one JSON line on stdout (the shell's
    # ig_tool parses the last JSON line, same contract as edit_tool).
    if args.ig_cmd == "sync":
        summary = calibration.sync()
        print(json.dumps(summary))
        return 0 if summary.get("ok") else 1

    if args.ig_cmd == "overview":
        print(json.dumps(calibration.overview()))
        return 0

    if args.ig_cmd == "link":
        job = queue.get_job(args.job_id)
        if job is None:
            print(json.dumps({"ok": False, "error": f"no job {args.job_id}"}))
            return 2
        score_data = queue.read_checkpoint(job, "score", 1)
        if not score_data:
            print(json.dumps({"ok": False, "error": "job has no score checkpoint"}))
            return 2
        clips = score_data["clips"]
        if not 0 <= args.clip < len(clips):
            print(json.dumps({"ok": False, "error": f"clip index out of range (0..{len(clips) - 1})"}))
            return 2
        calibration.link_clip(
            args.job_id, args.clip, args.media_id, clips[args.clip],
            link_source=args.source,
            config_version=score_data.get("scoring_config_version", 1),
        )
        print(json.dumps({"ok": True, "linked": {"job_id": args.job_id, "clip": args.clip, "media_id": args.media_id}}))
        return 0

    if args.ig_cmd == "unlink":
        removed = calibration.unlink(args.media_id)
        print(json.dumps({"ok": True, "removed": removed}))
        return 0

    if args.ig_cmd == "reject":
        calibration.reject_match(args.media_id, args.job_id, args.clip)
        print(json.dumps({"ok": True}))
        return 0

    # Human/legacy commands.
    conn = instagram.load_connection()
    if args.ig_cmd in ("media", "pull") and conn is None:
        print("Not connected. Run: publikclip ig connect --app-id ... --app-secret ...", file=sys.stderr)
        return 2
    if conn is not None:
        conn = instagram.refresh_if_needed(conn)

    if args.ig_cmd == "media":
        for m in instagram.recent_media(conn):
            if m.get("media_product_type") == "REELS" or m.get("media_type") == "VIDEO":
                caption = (m.get("caption") or "")[:60].replace("\n", " ")
                print(f"{m['id']}  {m.get('timestamp', '')[:10]}  {caption}")
        return 0

    if args.ig_cmd == "pull":
        rows = calibration.tracked()
        if not rows:
            print("No linked clips yet. Post an exported clip, then: publikclip ig link ...")
            return 0
        for row in rows:
            if not row["ig_media_id"]:
                continue
            try:
                metrics = instagram.media_insights(conn, row["ig_media_id"])
            except instagram.IgError as err:
                print(f"{row['ig_media_id']}: {err}", file=sys.stderr)
                continue
            calibration.store_metrics(row["ig_media_id"], metrics)
            views = metrics.get("views")
            watch = metrics.get("ig_reels_avg_watch_time")
            print(
                f"{row['ig_media_id']}  score {row['score']:.0f} → views {views}, "
                f"avg watch {round(watch / 1000, 1) if watch else '?'}s"
            )
        return 0

    if args.ig_cmd == "report":
        print(json.dumps(calibration.report(args.metric), indent=2))
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="publikclip")
    # Global on purpose, and it must stay the only declaration: a subparser
    # that also defines --jsonl overwrites this one with its own default, so
    # `publikclip --jsonl <cmd>` silently loses the flag and the desktop
    # shell -- which invokes it exactly that way -- receives nothing.
    parser.add_argument("--jsonl", action="store_true", help="machine-readable progress on stdout")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="process a YouTube URL or local video file")
    p_run.add_argument("source")
    p_run.add_argument("--llm", choices=["gemini", "ollama"], default=None)
    p_run.add_argument("--captions", default=None, help="caption preset name")
    p_run.add_argument("--camera", choices=["cut", "pan", "locked"], default=None)
    p_run.add_argument(
        "--partial-ok", action="store_true",
        help="if the LLM runs out mid-scoring, keep the moments already scored",
    )
    p_run.set_defaults(fn=cmd_run)

    p_resume = sub.add_parser("resume", help="resume a job from its checkpoints")
    p_resume.add_argument("job_id")
    p_resume.add_argument("--llm", choices=["gemini", "ollama"], default=None)
    p_resume.add_argument("--captions", default=None, help="caption preset name")
    p_resume.add_argument("--camera", choices=["cut", "pan", "locked"], default=None)
    p_resume.add_argument(
        "--partial-ok", action="store_true",
        help="if the LLM runs out mid-scoring, keep the moments already scored",
    )
    p_resume.set_defaults(fn=cmd_resume)

    p_cap = sub.add_parser(
        "caption",
        help="captions only — the whole video back, with captions burned in",
    )
    p_cap.add_argument("source", nargs="?", help="video file to caption")
    p_cap.add_argument("-o", "--output", help="output path (default: the source with .captioned.mp4)")
    p_cap.add_argument("--preset", default="classic", help="caption style (see --list-presets)")
    p_cap.add_argument("--list-presets", action="store_true", help="print the available styles and exit")
    p_cap.add_argument(
        "--tags", action="store_true",
        help="also detect laughter/gasps for [laughs] tags (adds an audio-event pass)",
    )
    p_cap.add_argument("--ass-only", action="store_true", help="write the .ass subtitle file, do not burn it in")
    p_cap.set_defaults(fn=cmd_caption)

    p_del = sub.add_parser("delete", help="delete jobs and their files")
    p_del.add_argument("job_ids", nargs="+", help="job id(s) to delete")
    p_del.add_argument("--keep-files", action="store_true", help="forget the job but leave its directory on disk")
    p_del.set_defaults(fn=cmd_delete)

    p_jobs = sub.add_parser("jobs", help="list jobs")
    p_jobs.set_defaults(fn=cmd_jobs)

    p_sessions = sub.add_parser("sessions", help="jobs with their live process state")
    p_sessions.add_argument("--json", action="store_true", help="machine-readable")
    p_sessions.set_defaults(fn=cmd_sessions)

    p_edit = sub.add_parser("edit", help="per-clip editing (context / visuals / render)")
    edit_sub = p_edit.add_subparsers(dest="edit_cmd", required=True)
    p_ctx = edit_sub.add_parser("context")
    p_ctx.add_argument("job_id")
    p_ctx.add_argument("clip", type=int)
    p_sv = edit_sub.add_parser("suggest-visuals")
    p_sv.add_argument("job_id")
    p_sv.add_argument("clip", type=int)
    p_sv.add_argument("--prefer", choices=["pexels", "gemini"], default="pexels")
    p_rcl = edit_sub.add_parser("render-clip")
    p_rcl.add_argument("job_id")
    p_rcl.add_argument("clip", type=int)
    p_edit.set_defaults(fn=cmd_edit)

    p_ig = sub.add_parser("ig", help="Instagram feedback loop (your own Meta app)")
    ig_sub = p_ig.add_subparsers(dest="ig_cmd", required=True)
    p_connect = ig_sub.add_parser("connect", help="OAuth against your own Meta app")
    p_connect.add_argument("--app-id", required=True)
    p_connect.add_argument("--app-secret", required=True)
    ig_sub.add_parser("sync", help="one sync pass: media + thumbnails + insights ladder + auto-fit (JSON)")
    ig_sub.add_parser("overview", help="everything the Loop screen renders (JSON)")
    ig_sub.add_parser("media", help="list your recent Reels to link against")
    p_link = ig_sub.add_parser("link", help="link a rendered clip to a posted Reel (JSON)")
    p_link.add_argument("job_id")
    p_link.add_argument("clip", type=int)
    p_link.add_argument("media_id")
    p_link.add_argument("--source", default="manual", choices=["manual", "match_confirmed"])
    p_unlink = ig_sub.add_parser("unlink", help="remove a clip↔Reel link (JSON)")
    p_unlink.add_argument("media_id")
    p_reject = ig_sub.add_parser("reject", help="'not this' — never suggest this pair again (JSON)")
    p_reject.add_argument("media_id")
    p_reject.add_argument("job_id")
    p_reject.add_argument("clip", type=int)
    ig_sub.add_parser("pull", help="fetch metrics for every linked clip")
    p_report = ig_sub.add_parser("report", help="score-vs-outcome calibration report")
    p_report.add_argument("--metric", default="views")
    p_ig.set_defaults(fn=cmd_ig)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
