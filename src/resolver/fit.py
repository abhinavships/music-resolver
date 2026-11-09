"""Stage 4 planning: align a downbeat (or beat) with the main hit and cover the clip."""

from __future__ import annotations

import librosa
import numpy as np

from . import beats as beats_mod
from . import segment
from .beats import strong_beats  # noqa: F401  (re-exported for verify and tests)
from .config import Config
from .mixing import choose_role
from .schemas import ClipAnalysis, Loop, MixPlan

SR = 22050
HOP = 512


def _candidates(b: beats_mod.Beats, use_downbeats: bool) -> list[tuple[float, float]]:
    if use_downbeats:
        return list(zip(b.downbeats, b.downbeat_strengths))
    return list(zip(b.times, b.strengths))


def _extend(cands: list[tuple[float, float]], loop: Loop) -> list[tuple[float, float]]:
    length = loop.end - loop.start
    out = [c for c in cands if c[0] < loop.end]
    body = [c for c in cands if loop.start <= c[0] < loop.end]
    for k in range(1, loop.repeats + 1):
        out.extend((t + k * length, s) for t, s in body)
    return out


def _strongest_onset(env: np.ndarray | None, track_duration: float, clip_duration: float, sr: int) -> float:
    if env is None or env.size == 0:
        return 0.0
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=HOP)
    fits = times + clip_duration <= track_duration
    if not fits.any():
        fits = np.ones_like(times, dtype=bool)
    return float(times[np.argmax(np.where(fits, env, -np.inf))])


def plan_mix(
    clip: ClipAnalysis, track_path: str, y: np.ndarray, hit_time: float | None = None,
    sr: int = SR, cfg: Config | None = None, role: str | None = None,
    duck_db: float | None = None, music_lu_below_dialogue: float | None = None,
    prefer_start: float | None = None,
) -> MixPlan:
    """`hit_time` set by the caller is a user override (C11) and is always synced to.

    `prefer_start` is the retrieved segment's start (7.4): among aligned beats
    that cover the clip, the one putting the music start nearest it wins.
    """
    cfg = cfg or Config()
    track_duration = len(y) / sr
    dur = clip.duration
    warnings: list[str] = []
    b = beats_mod.track(y, sr, cfg.beats)
    warnings.extend(b.notes)

    # Decide whether to sync to a hit at all.
    if hit_time is None:
        if not clip.hits:
            warnings.append("no hit point found; no hit sync")
        elif clip.hits[0].score < cfg.hits.min_score:
            warnings.append(
                f"no clear hit (top score {clip.hits[0].score:.2f} < {cfg.hits.min_score}); no hit sync"
            )
        else:
            hit_time = clip.hits[0].time
        user_hit = False
    else:
        user_hit = True
    if hit_time is not None and not b.times:
        warnings.append("no beats tracked; no hit sync")
        hit_time = None
    if hit_time is not None and not user_hit and b.confidence < cfg.beats.min_confidence:
        warnings.append(f"beats unreliable (confidence {b.confidence:.2f}); no hit sync")
        hit_time = None

    # One tracked downbeat is not a bar grid; align to beats instead.
    use_downbeats = b.downbeats_tracked and len(b.downbeats) >= 2
    sync = "none" if hit_time is None else ("downbeat" if use_downbeats else "beat")
    cands = _candidates(b, use_downbeats)
    # Loop points need at least two bar lines; estimate them from beats if the engine gave fewer.
    bars = b.downbeats if len(b.downbeats) >= 2 else beats_mod.estimate_downbeats(
        b.times, b.strengths, cfg.beats.beats_per_bar)
    loop: Loop | None = None
    short = dur < cfg.fit.short_clip_s

    def choose(cands: list[tuple[float, float]], length: float) -> tuple[float, float] | None:
        valid = [c for c in cands if c[0] - hit_time >= 0 and c[0] - hit_time + dur <= length]
        if not valid:
            return None
        if prefer_start is not None:
            return min(valid, key=lambda c: (abs(c[0] - hit_time - prefer_start), -c[1]))
        return max(valid, key=lambda c: c[1])

    beat: float | None = None
    offset = delay = 0.0
    if hit_time is not None:
        best = choose(cands, track_duration)
        if best is None:
            later = [c for c in cands if c[0] >= hit_time]
            lead = max(later, key=lambda c: c[1])[0] - hit_time if later else 0.0
            bar = bars[1] - bars[0] if len(bars) >= 2 else 0.0
            # Long enough that a candidate at most one bar after the hit still covers the clip.
            loop = segment.choose_loop(
                bars, track_duration, lead + dur + bar + 0.1, cfg.fit.loop_crossfade_ms, b.times)
            if loop is not None:
                extended = _extend(cands, loop)
                best = choose(extended, loop.end + loop.repeats * (loop.end - loop.start))
        if best is None:
            early = [c for c in cands if c[0] <= hit_time]
            best = early[-1] if early else cands[0]
            loop = None
            warnings.append("track cannot cover the clip with the beat aligned; music starts late or ends early")
        beat = best[0]
        offset = max(0.0, beat - hit_time)
        delay = max(0.0, hit_time - beat)
    else:
        if short:
            # Stinger behaviour (C9): open on the track's strongest onset.
            offset = _strongest_onset(b.onset_env, track_duration, dur, sr)
        elif b.downbeats:
            # C7, T5: no sync, but still start on a bar line (the first one in the chosen segment).
            start = prefer_start or 0.0
            later = [d for d in b.downbeats if d >= start - 1e-3 and d + dur <= track_duration]
            offset = later[0] if later else b.downbeats[0]
        elif prefer_start:
            offset = prefer_start
        if offset + dur > track_duration:
            loop = segment.choose_loop(bars, track_duration, offset + dur + 0.1, cfg.fit.loop_crossfade_ms, b.times)

    if loop is not None:
        warnings.append(f"track looped: bars {loop.start:.2f}-{loop.end:.2f} s repeated {loop.repeats}x")
        available = loop.end + loop.repeats * (loop.end - loop.start)
    else:
        available = track_duration
    if available - offset + delay < dur - 1e-3:
        warnings.append(f"music ends {dur - (available - offset + delay):.2f} s before the clip")
    if dur > cfg.fit.long_clip_s:
        warnings.append(f"clip is {dur:.0f} s (over {cfg.fit.long_clip_s:.0f} s); one cue covers it all")

    role = role or choose_role(clip.speech_regions, dur, cfg.mix)
    if music_lu_below_dialogue is None:
        music_lu_below_dialogue = (
            cfg.mix.featured_lu_below_dialogue if role == "featured" else cfg.mix.underscore_lu_below_dialogue
        )
    fade_in = 0.0 if short else (cfg.fit.fade_in_s if offset > 0 and delay == 0 else 0.0)

    return MixPlan(
        clip_path=clip.path, track_path=track_path, clip_duration=dur,
        hit_time=hit_time, hit_sync=sync, downbeat_used=sync == "downbeat",
        beats_engine=b.engine, beat_confidence=round(b.confidence, 3),
        beat_time_in_track=None if beat is None else round(beat, 4),
        segment_start=round(offset, 4), track_offset=round(offset, 4), music_delay=round(delay, 4),
        loop=loop, speech_regions=clip.speech_regions,
        fade_in=fade_in, fade_out=min(cfg.fit.fade_out_max_s, dur / 10),
        mix_role=role, music_lu_below_dialogue=music_lu_below_dialogue,
        duck_db=cfg.mix.duck_db if duck_db is None else duck_db,
        target_lufs=cfg.mix.target_lufs, true_peak_dbtp=cfg.mix.true_peak_dbtp,
        attack_ms=cfg.mix.attack_ms, release_ms=cfg.mix.release_ms,
        warnings=warnings,
    )
