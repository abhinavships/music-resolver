"""Stage 5: measure the rendered result against the PRD's targets."""

from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf

from . import media
from .fit import HOP, strong_beats
from .schemas import Check, MixPlan, VerifyReport

MAX_HIT_OFFSET_S = 0.08
MIN_SPEECH_MARGIN_DB = 10.0
MAX_DURATION_DIFF_S = 0.1


def _rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def _speech_samples(audio: np.ndarray, plan: MixPlan, sr: int) -> np.ndarray:
    parts = [audio[int(r.start * sr):int(r.end * sr)] for r in plan.speech_regions]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=audio.dtype)


def verify(plan: MixPlan, artifacts: dict[str, str]) -> VerifyReport:
    checks: list[Check] = []
    original, sr = sf.read(artifacts["original"], dtype="float32")
    stem, _ = sf.read(artifacts["music_stem"], dtype="float32")
    mix, _ = sf.read(artifacts["mix"], dtype="float32")
    original, stem, mix = original.mean(axis=1), stem.mean(axis=1), mix

    video_dur = media.stream_duration(artifacts["video"], "video")
    audio_dur = media.stream_duration(artifacts["video"], "audio")
    if video_dur is not None and audio_dur is not None:
        diff = abs(video_dur - audio_dur)
        checks.append(Check(
            name="av_duration", passed=diff <= MAX_DURATION_DIFF_S, value=round(diff, 3),
            target=f"<= {MAX_DURATION_DIFF_S} s", detail=f"video {video_dur:.3f} s, audio {audio_dur:.3f} s",
        ))
    else:
        checks.append(Check(name="av_duration", passed=False, target="both streams present", detail="missing stream"))

    music_active = _rms_db(stem) > -60.0
    checks.append(Check(
        name="music_present", passed=music_active, value=round(_rms_db(stem), 1), target="> -60 dB",
    ))

    peak = float(np.max(np.abs(mix)))
    checks.append(Check(name="no_clipping", passed=peak <= 1.0, value=round(peak, 3), target="peak <= 1.0"))

    if plan.speech_regions:
        margin = _rms_db(_speech_samples(original, plan, sr)) - _rms_db(_speech_samples(stem, plan, sr))
        checks.append(Check(
            name="speech_masking", passed=margin >= MIN_SPEECH_MARGIN_DB, value=round(margin, 1),
            target=f">= {MIN_SPEECH_MARGIN_DB} dB", detail="clip audio minus music, inside speech regions",
        ))

    if plan.hit_time is not None and music_active:
        # Re-track beats on the placed music rather than trusting the plan.
        y22 = librosa.resample(stem, orig_sr=sr, target_sr=22050)
        beats = [t for t, _ in strong_beats(y22)]
        if beats:
            offset = min(abs(t - plan.hit_time) for t in beats)
            checks.append(Check(
                name="hit_offset", passed=offset <= MAX_HIT_OFFSET_S, value=round(offset, 3),
                target=f"<= {MAX_HIT_OFFSET_S} s", detail=f"hit at {plan.hit_time:.2f} s",
            ))
        else:
            checks.append(Check(
                name="hit_offset", passed=False, target=f"<= {MAX_HIT_OFFSET_S} s",
                detail="no beats tracked in the placed music",
            ))
    return VerifyReport(passed=all(c.passed for c in checks), checks=checks)
