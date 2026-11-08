"""Beats and downbeats: Beat This! when installed, librosa otherwise (F1, D4, T5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import BeatsConfig

HOP = 512

_beat_this_model = None


@dataclass
class Beats:
    times: list[float]
    strengths: list[float]
    downbeats: list[float]
    confidence: float
    engine: str
    # True only when the engine itself tracked downbeats. librosa's bar lines are
    # an accent-based estimate, good enough for loop points, not for hit sync.
    downbeats_tracked: bool = False
    downbeat_strengths: list[float] = field(default_factory=list)
    onset_env: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def onset_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)


def strong_beats(y: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """(time, onset strength) for each librosa-tracked beat."""
    env = onset_envelope(y, sr)
    _, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP, units="frames")
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    return [(float(t), float(env[min(f, len(env) - 1)])) for t, f in zip(times, frames)]


def _strength_at(env: np.ndarray, times: list[float], sr: int) -> list[float]:
    if not times or env.size == 0:
        return []
    frames = librosa.time_to_frames(np.asarray(times), sr=sr, hop_length=HOP)
    frames = np.clip(frames, 0, len(env) - 1)
    # Max over a +/- 2 frame window, since engines and onsets differ by a few ms.
    return [float(env[max(0, f - 2):f + 3].max()) for f in frames]


def pulse_confidence(env: np.ndarray, times: list[float], strengths: list[float], sr: int,
                     cfg: BeatsConfig = BeatsConfig()) -> float:
    """0..1: is there a beat worth syncing to?

    Pulse clarity (autocorrelation of the detrended onset curve at the beat
    period or twice it) times an attack factor. Onset strength is log-spectral
    flux in dB, so it does not depend on the track's level: a sustained pad
    sits near 0.1, noise near 1, drums well above 5. Without the attack factor
    a slowly swelling pad looks perfectly periodic.
    """
    if len(times) < 4 or env.size == 0 or not strengths:
        return 0.0
    attack = float(np.clip(
        (np.mean(strengths) - cfg.onset_floor_db) / (cfg.onset_full_db - cfg.onset_floor_db), 0.0, 1.0))
    if attack == 0.0:
        return 0.0
    frames_per_s = sr / HOP
    x = env - uniform_filter1d(env, size=max(3, int(frames_per_s)), mode="nearest")
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    if ac[0] <= 0:
        return 0.0
    period = float(np.median(np.diff(times))) * frames_per_s
    clarity = 0.0
    for lag in (period, 2 * period):
        lo, hi = max(1, int(np.floor(lag)) - 2), int(np.ceil(lag)) + 3
        if lo < len(ac):
            clarity = max(clarity, float(ac[lo:min(hi, len(ac))].max() / ac[0]))
    return float(np.clip(clarity, 0.0, 1.0)) * attack


def estimate_downbeats(times: list[float], strengths: list[float], beats_per_bar: int) -> list[float]:
    """Pick the bar phase whose beats are loudest on average."""
    if not times:
        return []
    phases = list(range(min(beats_per_bar, len(times))))
    # On a short track some phases give a single bar line; prefer ones that give a whole bar.
    whole = [p for p in phases if len(times[p::beats_per_bar]) >= 2]
    best = max(whole or phases, key=lambda p: np.mean(strengths[p::beats_per_bar]))
    return times[best::beats_per_bar]


def beat_this_available() -> bool:
    try:
        import beat_this  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _beat_this(y: np.ndarray, sr: int, cfg: BeatsConfig) -> tuple[list[float], list[float]]:
    global _beat_this_model
    from beat_this.inference import Audio2Beats

    if _beat_this_model is None:
        _beat_this_model = Audio2Beats(checkpoint_path=cfg.checkpoint, device=_device(), dbn=False)
    beats, downbeats = _beat_this_model(y.astype(np.float32), sr)
    return [float(t) for t in beats], [float(t) for t in downbeats]


def track(y: np.ndarray, sr: int, cfg: BeatsConfig = BeatsConfig()) -> Beats:
    env = onset_envelope(y, sr)
    notes: list[str] = []
    if cfg.engine == "beat_this":
        if beat_this_available():
            try:
                times, downbeats = _beat_this(y, sr, cfg)
                strengths = _strength_at(env, times, sr)
                return Beats(
                    times=times, strengths=strengths, downbeats=downbeats,
                    confidence=pulse_confidence(env, times, strengths, sr, cfg), engine="beat_this",
                    downbeats_tracked=bool(downbeats),
                    downbeat_strengths=_strength_at(env, downbeats, sr), onset_env=env,
                )
            except Exception as exc:  # model download or inference failure
                notes.append(f"Beat This! failed ({type(exc).__name__}: {exc}); used librosa beats, no downbeats")
        else:
            notes.append("Beat This! not installed; used librosa beats, no downbeats")

    pairs = strong_beats(y, sr)
    times = [t for t, _ in pairs]
    strengths = [s for _, s in pairs]
    downbeats = estimate_downbeats(times, strengths, cfg.beats_per_bar)
    return Beats(
        times=times, strengths=strengths, downbeats=downbeats,
        downbeat_strengths=_strength_at(env, downbeats, sr), onset_env=env,
        confidence=pulse_confidence(env, times, strengths, sr, cfg), engine="librosa", notes=notes,
    )
