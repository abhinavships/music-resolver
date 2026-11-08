"""Speech regions: Silero VAD when installed, the energy-band detector otherwise (C1, C2)."""

from __future__ import annotations

import warnings

import librosa
import numpy as np

from .config import VadConfig
from .schemas import Region

HOP = 512
SILERO_SR = 16000

_silero_model = None


def _merge(regions: list[tuple[float, float]], min_len: float, max_gap: float) -> list[Region]:
    merged: list[list[float]] = []
    for a, b in sorted(regions):
        if merged and a - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [Region(start=round(a, 3), end=round(b, 3)) for a, b in merged if b - a >= min_len]


def energy_regions(y: np.ndarray, sr: int, cfg: VadConfig = VadConfig()) -> list[Region]:
    """Energy in the 300-3400 Hz band above an adaptive threshold.

    A stand-in for Silero VAD: it also fires on loud non-speech in that band.
    """
    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        return []
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = spec[(freqs >= 300) & (freqs <= 3400)].sum(axis=0)
    level = librosa.amplitude_to_db(band + 1e-9, ref=np.max)
    floor = np.percentile(level, 10)
    threshold = max(floor + 12.0, -35.0)
    active = level > threshold
    times = librosa.frames_to_time(np.arange(len(active)), sr=sr, hop_length=HOP)
    frames = [(float(t), float(t)) for t, on in zip(times, active) if on]
    return _merge(frames, cfg.min_region_s, cfg.max_gap_s)


def silero_available() -> bool:
    try:
        import silero_vad  # noqa: F401
    except Exception:
        return False
    return True


def silero_regions(y: np.ndarray, sr: int, cfg: VadConfig = VadConfig()) -> list[Region]:
    global _silero_model
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if y.size == 0:
        return []
    if _silero_model is None:
        with warnings.catch_warnings():
            # silero-vad loads a TorchScript model; torch warns that jit.load is deprecated.
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
            _silero_model = load_silero_vad()
    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SILERO_SR)
    stamps = get_speech_timestamps(
        torch.from_numpy(y16), _silero_model, threshold=cfg.silero_threshold,
        sampling_rate=SILERO_SR, return_seconds=True,
        speech_pad_ms=int(cfg.silero_pad_ms),
    )
    return _merge([(float(s["start"]), float(s["end"])) for s in stamps], cfg.min_region_s, cfg.max_gap_s)


def speech_regions(y: np.ndarray, sr: int, cfg: VadConfig = VadConfig()) -> tuple[list[Region], str]:
    """Return (regions, engine used). VAD output is accepted as is, so shouting
    and laughter count as dialogue for ducking (C2)."""
    if cfg.engine == "silero" and silero_available():
        return silero_regions(y, sr, cfg), "silero"
    return energy_regions(y, sr, cfg), "energy-band"
