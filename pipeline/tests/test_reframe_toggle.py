"""Reframing off: an already-vertical source is rendered as shot.

The camera director exists to crop a 9:16 window out of a wider frame. Given a
source already shot 9:16 its best possible output is the whole frame unchanged,
bought with face detection and active-speaker analysis on every finalist clip.
"""

from publikclip_pipeline import config
from publikclip_pipeline.camera.stage import CameraStage
from publikclip_pipeline.render import renderer


class _Ctx:
    """Minimal StageContext stand-in: the bypass must not touch anything else."""

    def __init__(self, reframe: bool):
        self.settings = config.Settings(reframe=reframe)
        self.prior = {}
        self.job_dir = None
        self.emitted = []

    def emit(self, fraction, message, stage=""):
        self.emitted.append((fraction, message))


class _Done:
    returncode = 0
    stderr = ""
    stdout = ""


def _capture_ffmpeg(monkeypatch):
    """Intercept only the render invocation.

    subprocess is a shared module object, so patching its `run` patches it for
    everyone — including ffmpeg_bin's capability probe, which resolves the
    binary lazily. Anything without a filter chain is passed through.
    """
    seen = {}
    real_run = renderer.subprocess.run

    def fake_run(args, **kwargs):
        argv = list(args)
        if "-vf" in argv:
            seen["args"] = argv
            return _Done()
        return real_run(args, **kwargs)

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    return seen


def _vf(seen):
    args = seen["args"]
    return args[args.index("-vf") + 1]


def test_settings_round_trip():
    s = config.Settings(reframe=False)
    assert config.Settings.from_json(s.to_json()).reframe is False


def test_settings_default_on_json_without_the_key():
    """Job dirs written before this existed must keep reframing."""
    assert config.Settings.from_json({}).reframe is True


def test_camera_stage_skips_everything_when_off():
    ctx = _Ctx(reframe=False)
    out = CameraStage().run(ctx)
    assert out["trajectories"] == {}
    assert out["reframe"] is False


def test_camera_stage_bypass_needs_no_prior_stages():
    """It returns before the ingest/diarize/events/score requirement, so the
    vision models are never even downloaded."""
    ctx = _Ctx(reframe=False)
    ctx.prior = None
    CameraStage().run(ctx)  # must not raise StageError


def test_toggling_reframe_invalidates_the_cached_camera_pass():
    stage = CameraStage()
    off = stage.run(_Ctx(reframe=False))
    assert stage.artifacts_ok(_Ctx(reframe=False), off) is True
    assert stage.artifacts_ok(_Ctx(reframe=True), off) is False


def test_old_checkpoint_without_the_key_still_reads_as_reframed():
    stage = CameraStage()
    legacy = {"trajectories": {}, "camera_settings": config.CameraSettings().__dict__}
    assert stage.artifacts_ok(_Ctx(reframe=False), legacy) is False


def test_render_without_reframing_drops_crop_and_sendcmd(tmp_path, monkeypatch):
    seen = _capture_ffmpeg(monkeypatch)
    renderer.render_clip(
        "in.mp4", tmp_path / "out.mp4", 0.0, 10.0, None, None, None,
        src_w=1080, src_h=1920,
    )
    vf = _vf(seen)
    assert "crop" not in vf
    assert "sendcmd" not in vf
    assert f"scale={renderer.OUT_W}:{renderer.OUT_H}" in vf


def test_render_with_a_trajectory_still_crops(tmp_path, monkeypatch):
    seen = _capture_ffmpeg(monkeypatch)
    renderer.render_clip(
        "in.mp4", tmp_path / "out.mp4", 0.0, 10.0, {"fps": 25, "frames": []},
        None, None, src_w=1920, src_h=1080,
    )
    vf = _vf(seen)
    assert "crop" in vf
    assert "sendcmd" in vf


def test_no_stray_cmd_file_when_reframing_is_off(tmp_path, monkeypatch):
    """The sendcmd script is the crop's, and is neither written nor unlinked."""
    _capture_ffmpeg(monkeypatch)
    out = tmp_path / "out.mp4"
    renderer.render_clip("in.mp4", out, 0.0, 10.0, None, None, None,
                         src_w=1080, src_h=1920)
    assert not out.with_suffix(".cmd").exists()
