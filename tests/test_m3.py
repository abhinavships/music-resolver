"""M3 hardening: timing, loudness, ducking, looping, engines and degraded paths."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pyloudnorm
import pytest
import soundfile as sf

from conftest import HIT_TIME, SPEECH, SR, make_rich_track, make_track
from resolver import beats as beats_mod
from resolver import clip as clip_stage
from resolver import config as config_mod
from resolver import fit as fit_stage
from resolver import media, mixing, segment, vad
from resolver.render import render
from resolver.schemas import ClipAnalysis, HitPoint, Loop, Region
from resolver.verify import verify

ROOT = Path(__file__).resolve().parents[1]


# ---------- fixtures ----------

def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def _write_video(path: Path, n_frames: int, jump_frame: int | None, fps: int = 30) -> None:
    """Static grid with a square that jumps sideways at `jump_frame`."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    x = 100
    for i in range(n_frames):
        frame = np.full((180, 320, 3), 40, dtype=np.uint8)
        frame[::20, :] = 90
        frame[:, ::20] = 90
        if jump_frame is not None and i >= jump_frame:
            x = 220
        cv2.rectangle(frame, (x, 60), (x + 50, 110), (230, 230, 230), -1)
        writer.write(frame)
    writer.release()


def _write_impact_audio(path: Path, duration: float, impact: float | None, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    audio = 0.002 * rng.standard_normal(n)
    if impact is not None:
        s, length = int(impact * SR), int(0.05 * SR)
        audio[s:s + length] += 0.9 * rng.standard_normal(length) * np.exp(-np.arange(length) / (0.01 * SR))
    sf.write(path, audio.astype(np.float32), SR)


def _load22(path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        return media.load_audio(path, fit_stage.SR, Path(tmp))


def _scale_to_lufs(path: Path, out: Path, target: float) -> None:
    """Re-master a track to `target` LUFS, soft-limiting peaks like a loud master."""
    y, sr = sf.read(path, dtype="float64")
    meter = pyloudnorm.Meter(sr)
    for _ in range(6):
        y = y * 10 ** ((target - meter.integrated_loudness(y)) / 20)
        if np.max(np.abs(y)) > 0.99:
            y = 0.99 * np.tanh(y / 0.99)
    sf.write(out, y.astype(np.float32), sr)


@pytest.fixture(scope="module")
def analysis(clip_path) -> ClipAnalysis:
    return clip_stage.analyse(clip_path)


@pytest.fixture(scope="module")
def no_hit_clip(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("nohit")
    video, wav, out = tmp / "v.mp4", tmp / "a.wav", tmp / "nohit.mp4"
    _write_video(video, 12 * 30, None)
    _write_impact_audio(wav, 12.0, None)
    media.mux(video, wav, out)
    return out


# ---------- C5, C6: timing ----------

def test_c5_vfr_hit_within_100ms(tmp_path):
    # 150 frames at 30 fps, then frames every 1/15 s; the jump is frame 240.
    cfr, vfr = tmp_path / "cfr.mp4", tmp_path / "vfr.mp4"
    _write_video(cfr, 300, jump_frame=240)
    _ffmpeg("-i", str(cfr), "-vf", "setpts='if(lt(N,150),N/30,5+(N-150)/15)/TB'",
            "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vfr))
    truth = 5.0 + (240 - 150) / 15  # 11.0 s
    nominal = cv2.VideoCapture(str(vfr)).get(cv2.CAP_PROP_FPS)
    assert abs(240 / nominal - truth) > 0.5, "fixture is not variable frame rate"

    result = clip_stage.analyse(vfr)
    assert not result.has_audio
    assert abs(result.hits[0].time - truth) < 0.1, result.hits


@pytest.mark.parametrize("layout", ["video_starts_late", "whole_file_offset"])
def test_c6_start_time_offset(tmp_path, layout):
    video, wav, clip = tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "clip.mp4"
    track_path = tmp_path / "rich.wav"
    make_rich_track(track_path)
    _write_impact_audio(wav, 12.0, impact=HIT_TIME)
    if layout == "video_starts_late":
        # Video stream starts 0.5 s after the audio; the jump is at 7.0 s on the shared timeline.
        _write_video(video, 12 * 30, jump_frame=int((HIT_TIME - 0.5) * 30))
        _ffmpeg("-itsoffset", "0.5", "-i", str(video), "-i", str(wav),
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(clip))
        expected_offset = 0.5
    else:
        # Both streams start 1 s into the container's timeline.
        synced = tmp_path / "synced.mp4"
        _write_video(video, 12 * 30, jump_frame=int(HIT_TIME * 30))
        media.mux(video, wav, synced)
        _ffmpeg("-itsoffset", "1.0", "-i", str(synced), "-map", "0", "-c", "copy", str(clip))
        expected_offset = None
    assert media.format_start(clip) > 0.9 or media.stream_offset(clip, "video") > 0.4

    result = clip_stage.analyse(clip)
    if expected_offset is not None:
        assert result.video_offset == pytest.approx(expected_offset, abs=0.03)
    top = result.hits[0]
    # Audio and motion agree only if both are on the same timeline.
    assert abs(top.time - HIT_TIME) < 0.1, result.hits
    assert top.audio_score > 0.5 and top.motion_score > 0.5, top

    plan = fit_stage.plan_mix(result, str(track_path), _load22(track_path))
    plan, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    report = verify(plan, artifacts)
    assert report.passed, report.model_dump_json(indent=2)
    # The mux keeps the source's audio/video relationship.
    def gap(path):
        info = media.probe(path)
        starts = {s["codec_type"]: float(s["start_time"]) for s in info["streams"]}
        return starts["video"] - starts["audio"]
    assert gap(artifacts["video"]) == pytest.approx(gap(clip), abs=0.05)


# ---------- T1, F2, F3, F6: loudness ----------

def _music_lufs(artifacts, start: float, end: float) -> float:
    stem, sr = sf.read(artifacts["music_stem"], dtype="float64")
    return float(pyloudnorm.Meter(sr).integrated_loudness(stem[int(start * sr):int(end * sr)]))


def test_t1_level_independent_of_master(analysis, tmp_path):
    base = tmp_path / "rich.wav"
    make_rich_track(base)
    levels = []
    for target in (-30.0, -8.0):
        track = tmp_path / f"master{int(target)}.wav"
        _scale_to_lufs(base, track, target)
        plan = fit_stage.plan_mix(analysis, str(track), _load22(track))
        plan, artifacts = render(plan, tmp_path / f"out{int(target)}.mp4", tmp_path / f"stems{int(target)}")
        assert verify(plan, artifacts).passed
        levels.append(_music_lufs(artifacts, 7.0, 11.0))
    assert abs(levels[0] - levels[1]) <= 1.0, levels


@pytest.mark.parametrize("role,below", [("underscore", 12.0), ("featured", 4.0)])
def test_f2_f3_role_sets_music_below_dialogue(analysis, tmp_path, role, below):
    track = tmp_path / "rich.wav"
    make_rich_track(track)
    # Lift the peak ceiling so the final limiter cannot move the balance; what is
    # measured is the role level itself.
    cfg = config_mod.Config.model_validate({"mix": {"true_peak_dbtp": 20.0}})
    plan = fit_stage.plan_mix(analysis, str(track), _load22(track), role=role, cfg=cfg)
    assert plan.music_lu_below_dialogue == below
    plan, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems", cfg)
    original, sr = sf.read(artifacts["original"], dtype="float64")
    meter = pyloudnorm.Meter(sr)
    dialogue = meter.integrated_loudness(mixing.region_samples(original, plan.speech_regions, sr))
    # Away from speech (and its 300 ms release) the music sits at its role level.
    music = _music_lufs(artifacts, 7.0, 11.0)
    assert dialogue - music == pytest.approx(below, abs=1.0), (dialogue, music)
    assert plan.final_balance_lu == pytest.approx(below, abs=1.0)


def test_f6_limiting_that_moves_the_balance_is_reported(analysis, tmp_path):
    track = tmp_path / "rich.wav"
    make_rich_track(track)
    plan = fit_stage.plan_mix(analysis, str(track), _load22(track), role="underscore")
    plan, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    assert plan.final_balance_lu is not None
    moved = abs(plan.final_balance_lu - plan.music_lu_below_dialogue)
    assert (moved > 2.0) == any("moved the balance" in w for w in plan.warnings)
    assert verify(plan, artifacts).passed


def test_role_follows_amount_of_speech():
    cfg = config_mod.MixConfig()
    assert mixing.choose_role([Region(start=0, end=6)], 10.0, cfg) == "underscore"
    assert mixing.choose_role([Region(start=0, end=1)], 10.0, cfg) == "featured"
    assert mixing.choose_role([], 10.0, cfg) == "featured"


def test_f6_final_lufs_and_peak(analysis, tmp_path):
    base, loud = tmp_path / "rich.wav", tmp_path / "loud.wav"
    make_rich_track(base)
    _scale_to_lufs(base, loud, -8.0)
    plan = fit_stage.plan_mix(analysis, str(loud), _load22(loud), role="featured")
    plan, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    mix, sr = sf.read(artifacts["mix"], dtype="float64")
    assert pyloudnorm.Meter(sr).integrated_loudness(mix) == pytest.approx(-14.0, abs=1.0)
    assert mixing.true_peak_db(mix) <= -1.0 + 1e-3
    report = verify(plan, artifacts)
    names = {c.name: c for c in report.checks}
    assert names["integrated_lufs"].passed and names["true_peak"].passed
    # Stems carry the same final gain, so they still sum to the mix.
    original, _ = sf.read(artifacts["original"], dtype="float64")
    stem, _ = sf.read(artifacts["music_stem"], dtype="float64")
    assert np.max(np.abs(original + stem - mix)) < 1e-5


def test_limiter_holds_ceiling_on_a_spike():
    sr = 48000
    x = np.zeros((sr, 2))
    x[sr // 2] = 0.9
    x += 0.05 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)[:, None]
    g = mixing.limiter_gain(x * 4, -1.0, lookahead=int(0.005 * sr))
    assert mixing.true_peak_db(x * 4 * g[:, None]) <= -1.0 + 0.05
    assert g[: sr // 4].min() > 0.99  # far from the spike, nothing changes


# ---------- F4: ducking ----------

def test_f4_duck_attack_release():
    sr = 48000
    env = mixing.duck_envelope_db(6 * sr, sr, [Region(start=2.0, end=4.0)], [-20.0], 80.0, 300.0)
    at = lambda t: env[int(round(t * sr))]  # noqa: E731
    assert at(1.90) == 0.0                         # before the attack
    assert at(1.96) == pytest.approx(-10.0, abs=0.1)  # halfway down
    assert at(2.00) == pytest.approx(-20.0)        # already down when speech starts
    assert at(4.00) == pytest.approx(-20.0)
    assert at(4.15) == pytest.approx(-10.0, abs=0.1)  # halfway up, release is slower
    assert at(4.31) == 0.0


def test_duck_deepens_to_keep_speech_margin(analysis, track_path, tmp_path):
    # A shallow configured duck still yields the 10 dB margin, because each region
    # is ducked as far as it needs.
    plan = fit_stage.plan_mix(analysis, str(track_path), _load22(track_path),
                              role="featured", music_lu_below_dialogue=0.0, duck_db=-1.0)
    plan, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    assert min(plan.region_duck_db) < -1.0
    masking = next(c for c in verify(plan, artifacts).checks if c.name == "speech_masking")
    assert masking.passed, masking


# ---------- T3: looping ----------

def test_t3_loop_whole_bars(tmp_path):
    # Clicks plus a sustained tone that is not locked to the bar: a hard cut at
    # a bar line lands at a random phase of the tone and clicks.
    sr = 48000
    path = tmp_path / "six.wav"
    make_track(path, duration=6.0, sr=sr)
    y, _ = sf.read(path, dtype="float32")
    y = y + 0.15 * np.sin(2 * np.pi * 201.0 * np.arange(len(y)) / sr).astype(np.float32)
    sf.write(path, y, sr)
    with tempfile.TemporaryDirectory() as tmp:
        y22 = media.load_audio(path, fit_stage.SR, Path(tmp))
    b = beats_mod.track(y22, fit_stage.SR)
    bars = b.downbeats if len(b.downbeats) >= 2 else beats_mod.estimate_downbeats(b.times, b.strengths, 4)
    loop = segment.choose_loop(bars, 6.0, 12.5, 30.0, b.times)
    assert loop is not None and loop.repeats >= 2
    # Whole bars: the loop starts on a bar line, ends near one, and spans a whole
    # number of beats of the true tempo (120 BPM), so joints stay on the grid.
    assert min(abs(loop.start - t) for t in bars) < 1e-3
    assert min(abs(loop.end - t) for t in bars) < 0.05
    beats_in_loop = (loop.end - loop.start) / 0.5
    assert abs(beats_in_loop - round(beats_in_loop)) < 0.04 and round(beats_in_loop) >= 2

    out = segment.build_looped(y, sr, loop)
    assert len(out) / sr >= 12.5
    s0, e0 = int(round(loop.start * sr)), int(round(loop.end * sr))
    margin = 2 * sr // 1000

    def max_jump(signal: np.ndarray, joint: int) -> float:
        return float(np.max(np.abs(np.diff(signal[joint - margin: joint + int(0.03 * sr) + margin]))))

    length = loop.end - loop.start
    for k in range(1, loop.repeats + 1):
        assert max_jump(out, int(round((loop.end + (k - 1) * length) * sr))) < 0.05
    hard_cut = np.concatenate([y[:e0], y[s0:e0]])
    assert max_jump(hard_cut, e0) > 0.05, "fixture would not reveal a click"


def test_t3_short_track_covers_a_longer_clip(analysis, tmp_path):
    track = tmp_path / "six.wav"
    make_rich_track(track, duration=6.0)
    plan = fit_stage.plan_mix(analysis, str(track), _load22(track))
    assert plan.loop is not None
    assert not any("ends" in w for w in plan.warnings), plan.warnings
    plan, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems")
    report = verify(plan, artifacts)
    assert report.passed, report.model_dump_json(indent=2)
    stem, sr = sf.read(artifacts["music_stem"], dtype="float32")
    # Music is present in the last two seconds, well past the 6 s track.
    assert mixing.rms_db(stem[int(9.5 * sr):int(11.0 * sr)].mean(axis=1)) > -50


def test_beat_period_fit_beats_jitter():
    rng = np.random.default_rng(1)
    times = list(0.3 + 0.5 * np.arange(40) + rng.uniform(-0.023, 0.023, 40))
    assert segment.beat_period(times) == pytest.approx(0.5, abs=0.002)


def test_looped_times_follow_the_loop():
    loop = Loop(start=1.0, end=3.0, repeats=2, crossfade_ms=30)
    assert segment.looped_times([0.5, 1.0, 2.0, 3.0, 4.0], loop) == [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# ---------- F1, D4: beat engines ----------

def test_d4_librosa_fallback_when_beat_this_missing(analysis, track_path, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "beat_this", None)  # import now raises ImportError
    assert not beats_mod.beat_this_available()
    b = beats_mod.track(_load22(track_path), fit_stage.SR)
    assert b.engine == "librosa" and not b.downbeats_tracked
    assert any("not installed" in n for n in b.notes)

    plan = fit_stage.plan_mix(analysis, str(track_path), _load22(track_path))
    assert plan.beats_engine == "librosa" and plan.hit_sync == "beat" and not plan.downbeat_used
    assert any("Beat This! not installed" in w for w in plan.warnings)
    report = verify(*render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    assert report.passed, report.model_dump_json(indent=2)


@pytest.mark.skipif(not beats_mod.beat_this_available(), reason="beat-this not installed")
def test_f1_downbeat_alignment(analysis, tmp_path):
    track = tmp_path / "rich.wav"
    make_rich_track(track)
    y = _load22(track)
    b = beats_mod.track(y, fit_stage.SR)
    assert b.engine == "beat_this" and b.downbeats_tracked
    plan = fit_stage.plan_mix(analysis, str(track), y)
    assert plan.hit_sync == "downbeat" and plan.downbeat_used
    assert min(abs(plan.beat_time_in_track - d) for d in b.downbeats) < 1e-3
    placed = plan.beat_time_in_track - plan.track_offset + plan.music_delay
    assert abs(placed - plan.hit_time) < 0.001
    report = verify(*render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    hit = next(c for c in report.checks if c.name == "hit_offset")
    assert hit.passed and "downbeat" in hit.detail, hit


def test_t5_unreliable_beats_skip_hit_sync(analysis, tmp_path):
    # A sustained pad with no attacks.
    sr = 48000
    t = np.arange(20 * sr) / sr
    pad = 0.2 * (np.sin(2 * np.pi * 196 * t) + np.sin(2 * np.pi * 294 * t)) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * t))
    path = tmp_path / "pad.wav"
    sf.write(path, pad.astype(np.float32), sr)
    plan = fit_stage.plan_mix(analysis, str(path), _load22(path))
    assert plan.hit_sync == "none", (plan.beat_confidence, plan.warnings)
    report = verify(*render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    assert any(s.startswith("hit_offset") for s in report.skipped)


# ---------- VAD ----------

def test_vad_falls_back_to_energy_when_silero_missing(clip_path, monkeypatch):
    monkeypatch.setattr(vad, "silero_available", lambda: False)
    with tempfile.TemporaryDirectory() as tmp:
        y = media.load_audio(clip_path, clip_stage.SR, Path(tmp))
    regions, engine = vad.speech_regions(y, clip_stage.SR)
    assert engine == "energy-band" and regions


def test_c2_vad_engine_recorded(analysis):
    expected = "silero" if vad.silero_available() else "energy-band"
    assert analysis.speech_method == expected


# ---------- C7, C9, C10: clip edge cases ----------

def test_c7_no_clear_hit(no_hit_clip, track_path, tmp_path):
    result = clip_stage.analyse(no_hit_clip)
    assert not result.hits or result.hits[0].score < 0.35, result.hits
    y = _load22(track_path)
    plan = fit_stage.plan_mix(result, str(track_path), y)
    assert plan.hit_sync == "none" and plan.hit_time is None
    assert any("no clear hit" in w or "no hit point" in w for w in plan.warnings)
    # Starts on a bar line at 0 s.
    b = beats_mod.track(y, fit_stage.SR)
    assert plan.music_delay == 0.0
    assert min(abs(plan.track_offset - d) for d in b.downbeats) < 1e-3
    report = verify(*render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    assert report.passed, report.model_dump_json(indent=2)
    assert any(s.startswith("hit_offset: no hit sync") for s in report.skipped)


def test_c7_user_hit_overrides_low_score(no_hit_clip, track_path):
    result = clip_stage.analyse(no_hit_clip)
    plan = fit_stage.plan_mix(result, str(track_path), _load22(track_path), hit_time=5.0)
    assert plan.hit_sync != "none" and plan.hit_time == 5.0


def test_c9_short_clip_opens_on_strongest_onset(tmp_path, track_path):
    video, wav, clip = tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "short.mp4"
    _write_video(video, int(2.5 * 30), None)
    _write_impact_audio(wav, 2.5, None)
    media.mux(video, wav, clip)
    result = clip_stage.analyse(clip)
    plan = fit_stage.plan_mix(result, str(track_path), _load22(track_path))
    assert plan.fade_in == 0.0
    assert plan.hit_sync == "none"
    # The click track's accented clicks are at 0.5 + 2k s.
    accent = (plan.track_offset - 0.5) / 2.0
    assert abs(accent - round(accent)) * 2.0 < 0.06, plan.track_offset
    report = verify(*render(plan, tmp_path / "out.mp4", tmp_path / "stems"))
    assert report.passed, report.model_dump_json(indent=2)


def test_c9_short_clip_with_hit_has_no_fade_in(track_path):
    clip = ClipAnalysis(path="c.mp4", duration=2.5, has_audio=True,
                        hits=[HitPoint(time=1.2, score=0.9, audio_score=1.0, motion_score=0.8)])
    plan = fit_stage.plan_mix(clip, str(track_path), _load22(track_path))
    assert plan.fade_in == 0.0 and plan.hit_sync != "none"


def test_c10_long_clip_warns(track_path):
    clip = ClipAnalysis(path="c.mp4", duration=95.0, has_audio=True,
                        hits=[HitPoint(time=40.0, score=0.9, audio_score=1.0, motion_score=0.8)])
    plan = fit_stage.plan_mix(clip, str(track_path), _load22(track_path))
    assert any("one cue" in w for w in plan.warnings)
    assert plan.loop is not None  # 30 s track, 95 s clip


# ---------- D6 ----------

def test_d6_render_failure_writes_report(clip_path, track_path, tmp_path, monkeypatch):
    from resolver import cli

    def broken(*args, **kwargs):
        raise RuntimeError("ffmpeg failed: simulated encoder error")

    monkeypatch.setattr(media, "mux", broken)
    out = tmp_path / "run"
    assert cli.main(["fit", str(clip_path), str(track_path), "--out", str(out)]) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "render_failed"
    assert "simulated encoder error" in report["skipped"][0]
    assert (out / "mix.json").exists() and (out / "stems" / "mix.wav").exists()


def test_cli_role_and_level_overrides(clip_path, track_path, tmp_path):
    from resolver import cli

    out = tmp_path / "run"
    code = cli.main(["fit", str(clip_path), str(track_path), "--out", str(out), "--role", "featured"])
    assert code == 0
    plan = json.loads((out / "mix.json").read_text())
    assert plan["mix_role"] == "featured" and plan["music_lu_below_dialogue"] == 4.0
    assert plan["music_gain_db"] is not None  # resolved levels written back after render


# ---------- config ----------

def test_config_example_matches_defaults():
    with open(ROOT / "resolver.example.toml", "rb") as fh:
        example = tomllib.load(fh)
    assert config_mod.Config.model_validate(example) == config_mod.Config()
    # Every default is spelled out in the example, not just implied.
    defaults = config_mod.Config().model_dump()
    for section, values in defaults.items():
        assert set(example[section]) == set(values), section


def test_config_partial_file_keeps_defaults(tmp_path):
    path = tmp_path / "resolver.toml"
    path.write_text("[mix]\ntarget_lufs = -16.0\n")
    cfg = config_mod.load(path)
    assert cfg.mix.target_lufs == -16.0
    assert cfg.mix.true_peak_dbtp == -1.0 and cfg.hits.min_score == 0.35


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "resolver.toml"
    path.write_text("[mix]\ntarget_lufz = -16.0\n")
    with pytest.raises(Exception):
        config_mod.load(path)


def test_config_drives_the_mix(analysis, track_path, tmp_path):
    cfg = config_mod.Config.model_validate({"mix": {"target_lufs": -18.0}})
    plan = fit_stage.plan_mix(analysis, str(track_path), _load22(track_path), cfg=cfg)
    plan, artifacts = render(plan, tmp_path / "out.mp4", tmp_path / "stems", cfg)
    report = verify(plan, artifacts, cfg)
    assert report.integrated_lufs == pytest.approx(-18.0, abs=1.0)
