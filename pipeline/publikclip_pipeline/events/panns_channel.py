"""PANNs Cnn14_DecisionLevelMax channel: framewise posteriors over the
AudioSet ontology, reduced to the event types the bus cares about.

Laughter-only decision (#2): crying classes are intentionally ABSENT from
the map — no battle-tested adult-crying model exists, and one false positive
would fire on three surfaces at once through the shared bus. PANNs' AudioSet
laughter classes stay in as the fusion partner for the jrgillick specialist.

Processes audio in 30 s chunks with 1 s overlap — Cnn14 on 2 h of audio at
once would need tens of GB of activations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from ..vendor.panns import models as panns_models

# AudioSet display_name → bus event type. Laughter classes fuse with the
# jrgillick channel; the rest are solo PANNs detections.
CLASS_MAP: dict[str, str] = {
    "Laughter": "laugh",
    "Giggle": "laugh",
    "Belly laugh": "laugh",
    "Chuckle, chortle": "laugh",
    "Snicker": "laugh",
    "Gasp": "gasp",
    "Screaming": "scream",
    "Shout": "shout",
    "Yell": "shout",
    "Applause": "applause",
    "Cheering": "cheer",
    "Clapping": "applause",
    # Gameplay audio. AudioSet separates gunfire by weapon; nothing downstream
    # uses that distinction, so they collapse to one bus type. Explosion stays
    # separate — in a shooter it marks a different kind of moment than fire.
    "Gunshot, gunfire": "gunfire",
    "Machine gun": "gunfire",
    "Fusillade": "gunfire",
    "Artillery fire": "gunfire",
    "Cap gun": "gunfire",
    "Explosion": "explosion",
}

CHUNK_SEC = 30.0
OVERLAP_SEC = 1.0

# Zero-shot PANNs posteriors on conversational audio run far below the
# dedicated-SED range the default DCASE thresholds assume: measured on a
# 2 h two-host comedy podcast, laughter tops out ~0.30 and gasps ~0.24
# while true negatives (applause/cheer in a studio) stay under 0.004.
# (enter, stay) per type; confidence is normalized by CONF_SCALE so a
# strong conversational laugh reads ~1.0 downstream.
THRESHOLDS: dict[str, tuple[float, float]] = {
    "laugh": (0.10, 0.05),
    "gasp": (0.10, 0.05),
    "scream": (0.15, 0.08),
    "shout": (0.15, 0.08),
    "applause": (0.15, 0.08),
    "cheer": (0.15, 0.08),
    # Measured over a 77.6 min solo FPS capture (1080x1920 60 fps, one
    # speaker). Gunfire separates cleanly from everything else in the mix:
    # peak 0.574, p99.9 0.455, p99 0.262, p95 0.076 — an order of magnitude
    # above the conversational classes on the same audio, none of which
    # reached even their own enter threshold (laughter topped 0.091, shout
    # 0.004). 0.15 sits between p95 and p99 and yields ~1.3 events/min, dense
    # enough for the gameplay channel's rolling baseline to mean something.
    "gunfire": (0.15, 0.08),
    # Explosion never fired on that capture: peak 0.075, below its own enter
    # threshold everywhere. Either the game has none, or PANNs hears them as
    # gunfire — the two are adjacent in the AudioSet ontology. Kept mapped
    # because it costs nothing (same forward pass) and a different game may
    # well separate them; treat it as unproven rather than calibrated.
    "explosion": (0.15, 0.08),
}
CONF_SCALE = 0.30

# Per-type overrides for that normalization. One scale for every class assumed
# they share a dynamic range, which conversational classes roughly do. Gunfire
# does not: it peaks at 0.574 on real game audio, so against 0.30 a third of
# its detections clip to exactly 1.0 and a distant shot reads the same as a
# point-blank burst. The gameplay channel is a density of *weighted* events, so
# that flattening lands squarely on the signal it needs. Scaled to its measured
# p99.99 instead, which keeps the top of the range at ~1.0 without clipping the
# body of the distribution into it.
CONF_SCALES: dict[str, float] = {"gunfire": 0.55}


def conf_scale(etype: str) -> float:
    return CONF_SCALES.get(etype, CONF_SCALE)


def load_class_indices() -> dict[int, str]:
    """AudioSet index → bus event type, for the classes we track."""
    csv_path = Path(__file__).parent.parent / "vendor" / "panns" / "class_labels_indices.csv"
    mapping: dict[int, str] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            name = row["display_name"].strip('"')
            if name in CLASS_MAP:
                mapping[int(row["index"])] = CLASS_MAP[name]
    return mapping


def framewise_probs(
    model: panns_models.Cnn14_DecisionLevelMax,
    y32k: np.ndarray,
    device: torch.device,
    progress=None,
) -> tuple[dict[str, np.ndarray], float]:
    """Per-event-type framewise posteriors on the model's 100 fps grid.
    Same-type classes collapse via max (a giggle IS a laugh)."""
    class_idx = load_class_indices()
    types = sorted(set(class_idx.values()))
    sr = panns_models.SAMPLE_RATE
    fps = panns_models.FRAMES_PER_SEC
    chunk = int(CHUNK_SEC * sr)
    overlap = int(OVERLAP_SEC * sr)
    total_frames = int(np.ceil(len(y32k) / sr * fps))
    out = {t: np.zeros(total_frames, dtype=np.float32) for t in types}

    pos = 0
    with torch.inference_mode():
        while pos < len(y32k):
            end = min(pos + chunk, len(y32k))
            seg = y32k[max(0, pos - overlap) : end]
            x = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(device)
            framewise = model(x)["framewise_output"][0].cpu().numpy()  # (frames, 527)
            lead_frames = int(round((pos - max(0, pos - overlap)) / sr * fps))
            frame0 = int(round(pos / sr * fps))
            usable = framewise[lead_frames:]
            n = min(len(usable), total_frames - frame0)
            for cls, etype in class_idx.items():
                np.maximum(
                    out[etype][frame0 : frame0 + n], usable[:n, cls], out=out[etype][frame0 : frame0 + n]
                )
            if progress:
                progress(end / len(y32k))
            pos = end
    return out, fps
