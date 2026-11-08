"""Extend a short track by looping whole bars (T3)."""

from __future__ import annotations

import math

import numpy as np

from .schemas import Loop


def beat_period(beat_times: list[float]) -> float | None:
    """Beat period from a straight-line fit through all beats.

    Single beat times jitter by a frame or two (about 23 ms with librosa);
    over a whole track the fit is far more precise than any one interval.
    """
    if len(beat_times) < 4:
        return None
    t = np.asarray(beat_times, dtype=float)
    rough = float(np.median(np.diff(t)))
    if rough <= 0:
        return None
    k = np.round((t - t[0]) / rough)
    slope, _ = np.polyfit(k, t, 1)
    return float(slope) if slope > 0 else None


def choose_loop(
    downbeats: list[float], track_duration: float, needed: float, crossfade_ms: float,
    beat_times: list[float] | None = None,
) -> Loop | None:
    """Loop from the first to the last downbeat (whole bars) until the track lasts `needed` seconds.

    With `beat_times`, the loop length is snapped to a whole number of fitted
    beat periods, so tracker jitter does not put a hiccup at every joint.
    Returns None if there is not at least one whole bar to loop.
    """
    bars = [t for t in downbeats if 0.0 <= t <= track_duration]
    if len(bars) < 2:
        return None
    start, end = bars[0], bars[-1]
    period = beat_period(beat_times) if beat_times else None
    if period:
        n = round((end - start) / period)
        if n >= 1:
            end = start + n * period
            while end > track_duration and n > 1:
                n -= 1
                end = start + n * period
    length = end - start
    if length <= 0 or end > track_duration:
        return None
    repeats = max(1, math.ceil((needed - end) / length))
    return Loop(start=round(start, 4), end=round(end, 4), repeats=repeats, crossfade_ms=crossfade_ms)


def looped_times(times: list[float], loop: Loop) -> list[float]:
    """Event times (beats, downbeats) on the looped track's timeline."""
    length = loop.end - loop.start
    out = [t for t in times if t < loop.end]
    body = [t for t in times if loop.start <= t < loop.end]
    for k in range(1, loop.repeats + 1):
        out.extend(t + k * length for t in body)
    return out


def build_looped(y: np.ndarray, sr: int, loop: Loop) -> np.ndarray:
    """Audio up to `loop.end`, then the bars [start, end) `repeats` times.

    Each repeat starts exactly on the bar line. Joints use an equal-power
    crossfade: the track's natural continuation past `end` fades out while the
    loop start fades in. If there is no audio past `end`, the audio before
    `start` fades in under the previous bar's tail instead.
    """
    s, e = int(round(loop.start * sr)), int(round(loop.end * sr))
    c = max(1, int(loop.crossfade_ms / 1000 * sr))
    body = y[s:e]
    if len(body) <= 2 * c:
        c = max(1, len(body) // 4)
    ramp = np.linspace(0.0, np.pi / 2, c, dtype=np.float32)
    fade_in, fade_out = np.sin(ramp), np.cos(ramp)
    if y.ndim == 2:
        fade_in, fade_out = fade_in[:, None], fade_out[:, None]

    parts = [y[:e].copy()]
    for _ in range(loop.repeats):
        seg = body.copy()
        if e + c <= len(y):
            seg[:c] = seg[:c] * fade_in + y[e:e + c] * fade_out
        elif s >= c:
            prev = parts[-1]
            prev[-c:] = prev[-c:] * fade_out + y[s - c:s] * fade_in
        else:
            seg[:c] *= fade_in
            parts[-1][-c:] *= fade_out
        parts.append(seg)
    return np.concatenate(parts)
