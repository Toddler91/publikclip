"""Captions-only tool: the source back whole, with captions burned in."""

import json
import wave

import numpy as np
import pytest

from publikclip_pipeline.captions import ass as ass_mod
from publikclip_pipeline.captions import tool as caption_tool


def _wav(path, seconds=1.0, sr=16000, channels=1):
    n = int(seconds * sr)
    samples = (np.sin(np.linspace(0, 200, n)) * 8000).astype("<i2")
    if channels > 1:
        samples = np.repeat(samples, channels)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


def test_reads_the_analysis_wav_without_librosa(tmp_path):
    """librosa.load pulls a lazy import chain that dies on speechbrain's k2
    proxy once alignment has run. This file is ours, so read it directly."""
    data = caption_tool._read_analysis_audio(_wav(tmp_path / "a.wav", seconds=0.5))
    assert len(data) == 8000
    assert data.dtype == np.float32
    assert -1.0 <= float(data.min()) and float(data.max()) < 1.0


def test_downmixes_a_stereo_analysis_wav(tmp_path):
    data = caption_tool._read_analysis_audio(_wav(tmp_path / "s.wav", seconds=0.5, channels=2))
    assert len(data) == 8000


def test_rejects_audio_that_is_not_16_bit(tmp_path):
    path = tmp_path / "b.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 100)
    with pytest.raises(caption_tool.CaptionError):
        caption_tool._read_analysis_audio(path)


def test_words_keep_absolute_times():
    """There is no clip to be relative to — the whole source is captioned."""
    segments = [{"words": [{"word": "hi", "start": 12.0, "end": 12.4}]}]
    words = caption_tool._words_for(segments)
    assert [(w.text, w.start, w.end) for w in words] == [("hi", 12.0, 12.4)]


def test_unaligned_words_are_dropped():
    """Alignment returns None bounds for a word it could not place; those would
    become a caption at time zero."""
    segments = [{"words": [
        {"word": "kept", "start": 1.0, "end": 1.2},
        {"word": "lost", "start": None, "end": None},
    ]}]
    assert [w.text for w in caption_tool._words_for(segments)] == ["kept"]


def test_missing_source_is_a_clear_error(tmp_path):
    with pytest.raises(caption_tool.CaptionError, match="No such file"):
        caption_tool.caption_video(tmp_path / "nope.mp4")


def test_unknown_preset_names_the_real_ones(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"")
    with pytest.raises(caption_tool.CaptionError, match="classic"):
        caption_tool.caption_video(src, preset="does-not-exist")


# --- the canvas override the tool needs -----------------------------------


def _words():
    return [ass_mod.Word(text="hello", start=0.0, end=0.5)]


def test_default_canvas_is_unchanged():
    """Every preset size is absolute in PlayRes units; the 9:16 default must
    render exactly as it did before the override existed."""
    doc = ass_mod.build_ass(_words(), [], preset_name="classic")
    assert f"PlayResX: {ass_mod.PLAY_RES_X}" in doc
    assert f"PlayResY: {ass_mod.PLAY_RES_Y}" in doc
    assert doc == ass_mod.build_ass(
        _words(), [], preset_name="classic",
        play_res=(ass_mod.PLAY_RES_X, ass_mod.PLAY_RES_Y),
    )


def test_canvas_override_scales_the_style_with_it():
    """A landscape canvas at the same font size would be huge; sizes scale."""
    tall = ass_mod.build_ass(_words(), [], preset_name="classic")
    wide = ass_mod.build_ass(_words(), [], preset_name="classic", play_res=(1920, 1080))
    assert "PlayResX: 1920" in wide and "PlayResY: 1080" in wide

    def font_size(doc):
        line = next(x for x in doc.splitlines() if x.startswith("Style: Cap,"))
        return int(line.split(",")[2])

    assert font_size(wide) < font_size(tall)


def test_every_preset_builds_on_an_overridden_canvas():
    for name in ass_mod.PRESETS:
        doc = ass_mod.build_ass(_words(), [], preset_name=name, play_res=(1080, 1350))
        assert "PlayResY: 1350" in doc
        assert "Style: Cap," in doc
