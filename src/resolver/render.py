"""Stage 4 render: mix music under the clip's own audio and mux with the original video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from . import media
from .fit import gain_envelope
from .schemas import MixPlan

SR = 48000


def build_music_stem(plan: MixPlan, track: np.ndarray, n: int, sr: int = SR) -> np.ndarray:
    """Place the track on the clip's timeline and apply gain, fades and ducking."""
    stem = np.zeros((n, 2), dtype=np.float32)
    start = int(plan.music_delay * sr)
    begin = int(plan.track_offset * sr)
    length = min(n - start, len(track) - begin)
    if length > 0:
        stem[start:start + length] = track[begin:begin + length]

    fade_out = int(plan.fade_out * sr)
    if fade_out:
        stem[n - fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    fade_in = int(plan.fade_in * sr)
    if fade_in and length > 0:
        stem[start:start + fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    return stem * gain_envelope(plan, n, sr)[:, None]


def render(plan: MixPlan, out: str | Path, work_dir: str | Path) -> dict[str, str]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    n = int(plan.clip_duration * SR)

    if media.has_audio(plan.clip_path):
        original = media.load_audio(plan.clip_path, SR, work, mono=False)
        padded = np.zeros((n, 2), dtype=np.float32)
        padded[:min(n, len(original))] = original[:n]
        original = padded
    else:
        original = np.zeros((n, 2), dtype=np.float32)

    track = media.load_audio(plan.track_path, SR, work, mono=False)
    stem = build_music_stem(plan, track, n)
    mix = original + stem

    # One gain for the whole mix, so clipping protection never changes the balance.
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    trim = 0.98 / peak if peak > 0.98 else 1.0
    paths = {
        "original": work / "original.wav",
        "music_stem": work / "music_stem.wav",
        "mix": work / "mix.wav",
    }
    sf.write(paths["original"], original * trim, SR, subtype="FLOAT")
    sf.write(paths["music_stem"], stem * trim, SR, subtype="FLOAT")
    sf.write(paths["mix"], mix * trim, SR, subtype="FLOAT")
    media.mux(plan.clip_path, paths["mix"], out)
    return {**{k: str(v) for k, v in paths.items()}, "video": str(out)}
