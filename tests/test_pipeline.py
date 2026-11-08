"""End-to-end checks against synthetic media with known hit, speech and beats."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import BPM, DURATION, HIT_TIME, SPEECH, make_rich_track, make_track
from resolver import clip as clip_stage
from resolver import fit as fit_stage
from resolver import media
from resolver.render import render
from resolver.schemas import ClipAnalysis, Region
from resolver.verify import verify


@pytest.fixture(scope="module")
def analysis(clip_path) -> ClipAnalysis:
    return clip_stage.analyse(clip_path)


def _load(path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        return media.load_audio(path, fit_stage.SR, Path(tmp))


def test_finds_the_main_hit(analysis):
    assert abs(analysis.hits[0].time - HIT_TIME) < 0.1


def test_speech_regions_cover_truth_without_the_silence(analysis):
    regions = analysis.speech_regions
    for a, b in SPEECH:
        overlap = sum(max(0.0, min(b, r.end) - max(a, r.start)) for r in regions)
        assert overlap >= 0.9 * (b - a)
    # Quiet stretch between 3.3 and 4.2 s must stay out, or ducking never recovers.
    assert not any(r.start < 4.2 and r.end > 3.3 for r in regions)


def test_plan_aligns_a_beat_with_the_hit(analysis, track_path):
    y = _load(track_path)
    plan = fit_stage.plan_mix(analysis, str(track_path), y)
    placed = plan.beat_time_in_track - plan.track_offset + plan.music_delay
    assert abs(placed - plan.hit_time) < 0.001
    assert not plan.warnings


def test_short_track_warns_instead_of_failing_silently(analysis, tmp_path):
    short = tmp_path / "short.wav"
    make_track(short, duration=5.0)
    plan = fit_stage.plan_mix(analysis, str(short), _load(short))
    assert plan.warnings


def test_envelope_ducks_inside_speech_and_recovers_outside():
    plan = fit_stage.MixPlan(
        clip_path="c", track_path="t", clip_duration=10.0, hit_time=None, beat_time_in_track=None,
        track_offset=0.0, music_delay=0.0, music_gain_db=-8.0, duck_db=-14.0,
        speech_regions=[Region(start=2.0, end=4.0)], fade_in=0.0, fade_out=0.0,
    )
    sr = 48000
    env = fit_stage.gain_envelope(plan, 10 * sr, sr)
    # duck_db is relative to the music level: -8 and -14 give -22 dB in speech.
    assert env[3 * sr] == pytest.approx(10 ** (-22 / 20), rel=1e-3)
    assert env[7 * sr] == pytest.approx(10 ** (-8 / 20), rel=1e-3)


def test_full_render_passes_verification(analysis, track_path, tmp_path):
    plan = fit_stage.plan_mix(analysis, str(track_path), _load(track_path), music_gain_db=-14.0, duck_db=-26.0)
    artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    report = verify(plan, artifacts)
    assert report.passed, report.model_dump_json(indent=2)
    assert abs(media.duration_of(artifacts["video"]) - DURATION) < 0.15


def test_verifier_catches_loud_music_under_speech(analysis, track_path, tmp_path):
    plan = fit_stage.plan_mix(analysis, str(track_path), _load(track_path), music_gain_db=0.0, duck_db=0.0)
    report = verify(plan, render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    masking = next(c for c in report.checks if c.name == "speech_masking")
    assert not masking.passed
    assert not report.passed


def test_verifier_catches_a_misaligned_beat(analysis, track_path, tmp_path):
    plan = fit_stage.plan_mix(analysis, str(track_path), _load(track_path), music_gain_db=-14.0, duck_db=-26.0)
    # Shift by a quarter of a beat, so the hit lands between beats.
    plan = plan.model_copy(update={"track_offset": plan.track_offset + 60.0 / BPM / 4})
    report = verify(plan, render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
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
    artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    assert media.has_audio(artifacts["video"])


def test_default_levels_pass_on_the_synthetic_clip(analysis, track_path, tmp_path):
    plan = fit_stage.plan_mix(analysis, str(track_path), _load(track_path))
    report = verify(plan, render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    assert report.passed, report.model_dump_json(indent=2)


@pytest.mark.parametrize("bpm", [92.0, 120.0, 148.0])
def test_alignment_holds_on_a_syncopated_track_with_a_pad(analysis, tmp_path, bpm):
    track = tmp_path / "rich.wav"
    make_rich_track(track, bpm=bpm)
    plan = fit_stage.plan_mix(analysis, str(track), _load(track))
    report = verify(plan, render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    hit = next(c for c in report.checks if c.name == "hit_offset")
    assert hit.passed, report.model_dump_json(indent=2)


def test_cli_writes_every_artifact(clip_path, track_path, tmp_path):
    from resolver.cli import main

    out = tmp_path / "run"
    code = main(["fit", str(clip_path), str(track_path), "--out", str(out)])
    assert code == 0
    for name in ("clip.json", "mix.json", "report.json", "output.mp4"):
        assert (out / name).exists()
