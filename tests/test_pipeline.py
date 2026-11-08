"""End-to-end checks against synthetic media with known hit, speech and beats."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import BPM, DURATION, HIT_TIME, SPEECH, make_rich_track, make_track
from resolver import clip as clip_stage
from resolver import fit as fit_stage
from resolver import media, vad
from resolver.render import render
from resolver.schemas import ClipAnalysis
from resolver.verify import verify


@pytest.fixture(scope="module")
def analysis(clip_path) -> ClipAnalysis:
    return clip_stage.analyse(clip_path)


def _load(path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        return media.load_audio(path, fit_stage.SR, Path(tmp))


def test_finds_the_main_hit(analysis):
    assert abs(analysis.hits[0].time - HIT_TIME) < 0.1


# Silero trims region ends on synthetic speech, so its bar is the spec's real-clip
# recall bar (M4); the energy detector keeps the original, stricter one.
@pytest.mark.parametrize("engine,recall", [("energy", 0.9), ("silero", 0.8)])
def test_c1_speech_regions_cover_truth_without_the_silence(clip_path, engine, recall):
    if engine == "silero" and not vad.silero_available():
        pytest.skip("silero-vad not installed")
    with tempfile.TemporaryDirectory() as tmp:
        y = media.load_audio(clip_path, clip_stage.SR, Path(tmp))
    fn = vad.silero_regions if engine == "silero" else vad.energy_regions
    regions = fn(y, clip_stage.SR)
    for a, b in SPEECH:
        overlap = sum(max(0.0, min(b, r.end) - max(a, r.start)) for r in regions)
        assert overlap >= recall * (b - a), regions
    # Quiet stretch between 3.3 and 4.2 s must stay out, or ducking never recovers.
    assert not any(r.start < 4.2 and r.end > 3.3 for r in regions)


def test_plan_aligns_a_beat_with_the_hit(analysis, track_path):
    y = _load(track_path)
    plan = fit_stage.plan_mix(analysis, str(track_path), y)
    placed = plan.beat_time_in_track - plan.track_offset + plan.music_delay
    assert abs(placed - plan.hit_time) < 0.001
    # The only acceptable note is which beat engine ran (D4).
    assert not [w for w in plan.warnings if "Beat This!" not in w]


def test_short_track_loops_and_says_so(analysis, tmp_path):
    short = tmp_path / "short.wav"
    make_track(short, duration=5.0)
    plan = fit_stage.plan_mix(analysis, str(short), _load(short))
    assert plan.loop is not None
    assert any("looped" in w for w in plan.warnings)
    placed = plan.beat_time_in_track - plan.track_offset + plan.music_delay
    assert abs(placed - plan.hit_time) < 0.001


def test_full_render_passes_verification(analysis, track_path, tmp_path):
    plan = fit_stage.plan_mix(analysis, str(track_path), _load(track_path))
    plan, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    report = verify(plan, artifacts)
    assert report.passed, report.model_dump_json(indent=2)
    assert abs(media.duration_of(artifacts["video"]) - DURATION) < 0.15


def test_verifier_catches_loud_music_under_speech(analysis, track_path, tmp_path):
    plan = fit_stage.plan_mix(
        analysis, str(track_path), _load(track_path), music_lu_below_dialogue=0.0, duck_db=0.0,
    )
    plan.enforce_speech_margin = False
    report = verify(*render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    masking = next(c for c in report.checks if c.name == "speech_masking")
    assert not masking.passed
    assert not report.passed


def test_verifier_catches_a_misaligned_beat(analysis, track_path, tmp_path):
    plan = fit_stage.plan_mix(analysis, str(track_path), _load(track_path))
    # Shift by a quarter of a beat, so the hit lands between beats.
    plan = plan.model_copy(update={"track_offset": plan.track_offset + 60.0 / BPM / 4})
    report = verify(*render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    hit = next(c for c in report.checks if c.name == "hit_offset")
    assert not hit.passed


def test_clip_without_audio_still_renders(track_path, tmp_path):
    import cv2

    silent = tmp_path / "silent.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
    for i in range(150):
        writer.write(np.full((90, 160, 3), (i * 2) % 255, dtype=np.uint8))
    writer.release()
    analysis = clip_stage.analyse(silent)
    assert not analysis.has_audio and analysis.speech_regions == []
    plan = fit_stage.plan_mix(analysis, str(track_path), _load(track_path))
    _, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    assert media.has_audio(artifacts["video"])


@pytest.mark.parametrize("bpm", [92.0, 120.0, 148.0])
def test_alignment_holds_on_a_syncopated_track_with_a_pad(analysis, tmp_path, bpm):
    track = tmp_path / "rich.wav"
    make_rich_track(track, bpm=bpm)
    plan = fit_stage.plan_mix(analysis, str(track), _load(track))
    report = verify(*render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    hit = next(c for c in report.checks if c.name == "hit_offset")
    assert hit.passed, report.model_dump_json(indent=2)


def test_cli_writes_every_artifact(clip_path, track_path, tmp_path):
    from resolver.cli import main

    out = tmp_path / "run"
    code = main(["fit", str(clip_path), str(track_path), "--out", str(out)])
    assert code == 0
    for name in ("clip.json", "mix.json", "report.json", "output.mp4"):
        assert (out / name).exists()
