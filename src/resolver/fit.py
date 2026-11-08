"""Stage 4 planning: align a strong beat with the main hit and fit the clip length."""

from __future__ import annotations

import librosa
import numpy as np

from .schemas import ClipAnalysis, MixPlan

SR = 22050
HOP = 512


def strong_beats(y: np.ndarray, sr: int = SR) -> list[tuple[float, float]]:
    """Return (time, strength) for each tracked beat."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    _, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP, units="frames")
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    return [(float(t), float(env[min(f, len(env) - 1)])) for t, f in zip(times, frames)]


def plan_mix(
    clip: ClipAnalysis, track_path: str, y: np.ndarray, hit_time: float | None = None,
    music_gain_db: float = -8.0, duck_db: float = -14.0, sr: int = SR,
) -> MixPlan:
    track_duration = len(y) / sr
    warnings: list[str] = []
    if hit_time is None and clip.hits:
        hit_time = clip.hits[0].time
    beats = strong_beats(y, sr)

    best: tuple[float, float] | None = None
    if hit_time is not None and beats:
        # Prefer a beat that lets the track cover the whole clip with no delay.
        valid = [
            b for b in beats
            if b[0] - hit_time >= 0 and b[0] - hit_time + clip.duration <= track_duration
        ]
        if valid:
            best = max(valid, key=lambda b: b[1])
        else:
            early = [b for b in beats if b[0] <= hit_time]
            best = early[-1] if early else beats[0]
            warnings.append("track cannot cover the clip with the beat aligned; music starts late or ends early")
    elif hit_time is None:
        warnings.append("no hit point found; track starts at 0")
    else:
        warnings.append("no beats tracked; track starts at 0")

    if best is None:
        offset, delay, beat = 0.0, 0.0, None
    else:
        beat = best[0]
        offset = max(0.0, beat - hit_time)
        delay = max(0.0, hit_time - beat)

    if track_duration - offset + delay < clip.duration:
        warnings.append(
            f"music ends {clip.duration - (track_duration - offset + delay):.2f} s before the clip"
        )

    return MixPlan(
        clip_path=clip.path, track_path=track_path, clip_duration=clip.duration,
        hit_time=hit_time, beat_time_in_track=beat, track_offset=round(offset, 4),
        music_delay=round(delay, 4), music_gain_db=music_gain_db, duck_db=duck_db,
        speech_regions=clip.speech_regions,
        fade_in=0.05 if offset > 0 and delay == 0 else 0.0,
        fade_out=min(0.5, clip.duration / 10), warnings=warnings,
    )


def gain_envelope(plan: MixPlan, n: int, sr: int, ramp: float = 0.08) -> np.ndarray:
    """Linear music gain over time: base level, lowered with short ramps during speech."""
    base = 10 ** (plan.music_gain_db / 20)
    duck = 10 ** (plan.duck_db / 20)
    mask = np.zeros(n, dtype=np.float32)
    for region in plan.speech_regions:
        mask[int(region.start * sr):int(region.end * sr)] = 1.0
    width = max(1, int(ramp * sr))
    # Dilate first so the music is already down when speech starts, then smooth.
    if mask.any():
        mask = np.convolve(mask, np.ones(2 * width), mode="same")
        mask = np.minimum(mask, 1.0)
        mask = np.convolve(mask, np.ones(width) / width, mode="same")
    return base * (1.0 - mask * (1.0 - duck))
