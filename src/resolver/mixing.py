"""Loudness-based levels, ducking and final loudness (T1, F2, F3, F4, F6).

Levels are set from measured loudness, not fixed gains, so the result does not
depend on how loud a track was mastered.
"""

from __future__ import annotations

import numpy as np
import pyloudnorm
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

from .config import MixConfig
from .schemas import MixPlan, Region

MIN_LOUDNESS_S = 0.4  # one BS.1770 gating block
TRUE_PEAK_OVERSAMPLE = 4
MAX_LOUDNESS_PASSES = 12


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def lufs(x: np.ndarray, sr: int) -> float | None:
    """Integrated loudness, or None if too short or silent."""
    if len(x) < int(MIN_LOUDNESS_S * sr):
        return None
    value = float(pyloudnorm.Meter(sr).integrated_loudness(x))
    return value if np.isfinite(value) else None


def true_peak_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    up = resample_poly(x, TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    return _db(float(np.max(np.abs(up))))


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)) if x.size else -180.0


def region_samples(x: np.ndarray, regions: list[Region], sr: int) -> np.ndarray:
    parts = [x[int(r.start * sr):int(r.end * sr)] for r in regions]
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros((0,) + x.shape[1:], dtype=x.dtype)
    return np.concatenate(parts)


def duck_envelope_db(
    n: int, sr: int, regions: list[Region], depths_db: list[float],
    attack_ms: float, release_ms: float,
) -> np.ndarray:
    """Gain in dB (<= 0). Each region is fully ducked from its start to its end:
    the ramp down finishes as speech starts (attack) and the ramp up begins as it ends (release)."""
    env = np.zeros(n, dtype=np.float64)
    t = np.arange(n) / sr
    attack, release = attack_ms / 1000.0, release_ms / 1000.0
    for region, depth in zip(regions, depths_db):
        lo = int(max(0.0, region.start - attack) * sr)
        hi = min(n, int(np.ceil((region.end + release) * sr)) + 1)
        if hi <= lo:
            continue
        tt = t[lo:hi]
        m = np.ones_like(tt)
        if attack > 0:
            m = np.where(tt < region.start, 1.0 - (region.start - tt) / attack, m)
        if release > 0:
            m = np.where(tt > region.end, 1.0 - (tt - region.end) / release, m)
        m = np.clip(m, 0.0, 1.0)
        env[lo:hi] = np.minimum(env[lo:hi], depth * m)
    return env


def limiter_gain(x: np.ndarray, ceiling_db: float, lookahead: int) -> np.ndarray:
    """Per-sample gain that keeps the oversampled peak under the ceiling.

    A minimum filter of width 2L+1 followed by a moving average of width L+1
    never exceeds the gain required at any sample, so no peak gets through,
    and the gain moves smoothly over about L samples either side of a peak.
    """
    n = len(x)
    up = np.abs(resample_poly(x, TRUE_PEAK_OVERSAMPLE, 1, axis=0))
    if up.ndim == 2:
        up = up.max(axis=1)
    up = up[: n * TRUE_PEAK_OVERSAMPLE]
    if len(up) < n * TRUE_PEAK_OVERSAMPLE:
        up = np.pad(up, (0, n * TRUE_PEAK_OVERSAMPLE - len(up)))
    # Include each sample's neighbouring inter-sample points.
    peak = up.reshape(n, TRUE_PEAK_OVERSAMPLE).max(axis=1)
    peak = np.maximum(peak, np.roll(peak, 1))
    ceiling = 10 ** (ceiling_db / 20)
    required = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if np.all(required >= 1.0):
        return np.ones(n)
    held = minimum_filter1d(required, size=2 * lookahead + 1, mode="nearest")
    return np.minimum(uniform_filter1d(held, size=lookahead + 1, mode="nearest"), required.max())


def _balance(original: np.ndarray, music: np.ndarray, regions: list[Region], sr: int, plan: MixPlan) -> float | None:
    """Dialogue loudness minus music loudness away from speech and its duck ramps."""
    if not regions:
        return None
    dialogue = lufs(region_samples(original, regions, sr), sr)
    keep = np.ones(len(music), dtype=bool)
    for r in regions:
        keep[int(max(0.0, r.start - plan.attack_ms / 1000) * sr):int((r.end + plan.release_ms / 1000) * sr)] = False
    keep &= np.abs(_mono(music)) > 0  # skip unplaced stretches
    away = lufs(music[keep], sr) if keep.any() else None
    if dialogue is None or away is None:
        return None
    return dialogue - away


def choose_role(regions: list[Region], duration: float, cfg: MixConfig) -> str:
    speech = sum(r.end - r.start for r in regions)
    return "featured" if duration <= 0 or speech / duration <= cfg.featured_max_speech_fraction else "underscore"


def mix(
    plan: MixPlan, original: np.ndarray, music: np.ndarray, sr: int, cfg: MixConfig = MixConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MixPlan]:
    """Level and duck the placed music under the clip audio, then set final loudness.

    `original` and `music` are (n, channels) on the clip timeline; `music` is
    placed and faded but not yet level-set. Returns the processed original,
    music stem and mix (the same final gain applied to all three, so stems sum
    to the mix) and the plan with the resolved levels filled in.
    """
    n = len(original)
    warnings = list(plan.warnings)
    original = original * 10 ** (plan.original_gain_db / 20)
    regions = plan.speech_regions

    dialogue = lufs(region_samples(original, regions, sr), sr) if regions else None
    reference = dialogue if dialogue is not None else cfg.no_speech_reference_lufs
    if regions and dialogue is None:
        warnings.append("speech too short or quiet to measure; music level set from the no-speech reference")
    music_target = reference - plan.music_lu_below_dialogue

    active = np.flatnonzero(np.abs(_mono(music)) > 1e-7)
    measured = lufs(music[active[0]:active[-1] + 1], sr) if active.size else None
    if measured is None:
        music_gain_db = 0.0
        if active.size:
            warnings.append("music too short to measure loudness; left at file level")
    else:
        music_gain_db = music_target - measured
    music = music * 10 ** (music_gain_db / 20)

    depths: list[float] = []
    orig_mono, music_mono = _mono(original), _mono(music)
    for region in regions:
        depth = plan.duck_db
        if plan.enforce_speech_margin:
            o = rms_db(region_samples(orig_mono, [region], sr))
            m = rms_db(region_samples(music_mono, [region], sr))
            needed = o - cfg.min_speech_margin_db - cfg.speech_margin_safety_db - m
            depth = min(depth, needed)
        depths.append(round(float(depth), 2))
    env = duck_envelope_db(n, sr, regions, depths, plan.attack_ms, plan.release_ms)
    music = music * (10 ** (env / 20))[:, None]

    total = original + music
    gain = np.ones(n)
    mix_gain_db = 0.0
    lookahead = max(1, int(cfg.limiter_lookahead_ms / 1000 * sr))
    # Normalise, limit, and correct the loudness lost to limiting; a loud
    # transient can take several rounds to settle.
    for _ in range(MAX_LOUDNESS_PASSES):
        loud = lufs(total * gain[:, None], sr)
        if loud is None:
            break
        step = plan.target_lufs - loud
        if abs(step) < 0.05 and true_peak_db(total * gain[:, None]) <= plan.true_peak_dbtp:
            break
        gain = gain * 10 ** (step / 20)
        gain = gain * limiter_gain(total * gain[:, None], plan.true_peak_dbtp - 0.1, lookahead)
    # Safety net: the oversampled peak must sit under the ceiling.
    over = true_peak_db(total * gain[:, None]) - plan.true_peak_dbtp
    if over > 0:
        gain = gain * 10 ** (-(over + 0.05) / 20)
    mix_gain_db = _db(float(np.median(gain)))
    reduction = _db(float(gain.min())) - mix_gain_db

    g = gain[:, None]
    balance = _balance(original * g, music * g, regions, sr, plan)
    if balance is not None and abs(balance - plan.music_lu_below_dialogue) > cfg.balance_warn_lu:
        warnings.append(
            f"final limiting moved the balance to {balance:.1f} LU (role target "
            f"{plan.music_lu_below_dialogue:.0f} LU); the clip's peaks are high for {plan.target_lufs} LUFS"
        )
    resolved = plan.model_copy(update={
        "dialogue_lufs": None if dialogue is None else round(dialogue, 2),
        "music_lufs": round(music_target, 2) if measured is not None else None,
        "music_gain_db": round(music_gain_db, 2),
        "region_duck_db": depths,
        "mix_gain_db": round(mix_gain_db, 2),
        "limiter_max_reduction_db": round(reduction, 2),
        "final_balance_lu": None if balance is None else round(balance, 2),
        "warnings": warnings,
    })
    return original * g, music * g, total * g, resolved
