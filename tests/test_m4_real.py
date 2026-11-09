"""M4: the real reel against hand labels. Skipped until fixtures/real/ has clip.mp4 and labels.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from resolver import clip as clip_stage
from resolver.config import Config

REAL = Path(__file__).resolve().parent.parent / "fixtures" / "real"
CLIP = REAL / "clip.mp4"
LABELS = REAL / "labels.json"

needs_real = pytest.mark.skipif(
    not (CLIP.exists() and LABELS.exists()),
    reason="fixtures/real/clip.mp4 and labels.json not present (see fixtures/real/README.md)",
)


def speech_recall(found: list[tuple[float, float]], labelled: list[tuple[float, float]], step: float = 0.01) -> float:
    """Fraction of labelled speech time covered by detected regions."""
    total = covered = 0
    for a, b in labelled:
        t = a
        while t < b:
            total += 1
            if any(s <= t < e for s, e in found):
                covered += 1
            t += step
    return covered / total if total else 1.0


def test_speech_recall_helper():
    assert speech_recall([(0, 1)], [(0, 1)]) == pytest.approx(1.0)
    assert speech_recall([(0, 0.5)], [(0, 1)]) == pytest.approx(0.5, abs=0.02)
    assert speech_recall([], []) == 1.0


@pytest.fixture(scope="module")
def real():
    labels = json.loads(LABELS.read_text())
    analysis = clip_stage.analyse(CLIP, Config())
    return labels, analysis


@needs_real
def test_real_clip_labels(real):
    labels, analysis = real
    assert analysis.hits, "no hits detected on the real clip"
    hit = analysis.hits[0].time
    assert abs(hit - labels["main_hit"]) <= 0.15, f"main hit {hit} s vs label {labels['main_hit']} s"
    found = [(r.start, r.end) for r in analysis.speech_regions]
    recall = speech_recall(found, [tuple(r) for r in labels["speech_regions"]])
    assert recall >= 0.8, f"speech recall {recall:.2f}"


@needs_real
def test_real_clip_main_hit_syncs(real):
    """The labelled hit must also clear `hits.min_score`, or fit would skip hit sync (C7)."""
    labels, analysis = real
    near = [h for h in analysis.hits if abs(h.time - labels["main_hit"]) <= 0.15]
    assert near and near[0].score >= Config().hits.min_score, (
        f"labelled hit scores {near[0].score if near else 'none'}; min_score {Config().hits.min_score}"
    )


@needs_real
def test_real_clip_licence_label():
    labels = json.loads(LABELS.read_text())
    assert labels.get("track_licence") in {"cc0", "cc-by", "cc-by-sa", "pd", "own", "cc-by-nc", "cc-by-nc-sa"}
    if labels["track_licence"].startswith("cc-by"):
        assert labels.get("track_attribution"), "CC BY needs an attribution line (L2)"
