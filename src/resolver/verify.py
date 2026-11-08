"""Stage 5: measure the rendered result against the targets."""

from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf

from . import media
from . import beats as beats_mod
from .config import Config
from .mixing import lufs, region_samples, rms_db, true_peak_db
from .schemas import Check, MixPlan, VerifyReport


def verify(plan: MixPlan, artifacts: dict[str, str], cfg: Config | None = None) -> VerifyReport:
    cfg = cfg or Config()
    v = cfg.verify
    checks: list[Check] = []
    skipped: list[str] = []
    original, sr = sf.read(artifacts["original"], dtype="float32")
    stem, _ = sf.read(artifacts["music_stem"], dtype="float32")
    mix, _ = sf.read(artifacts["mix"], dtype="float32")
    original_m, stem_m = original.mean(axis=1), stem.mean(axis=1)

    # Compare where each stream ends on the shared timeline, so a stream that
    # starts late in the source (C6) is not mistaken for a length mismatch.
    video_span = media.stream_span(artifacts["video"], "video")
    audio_span = media.stream_span(artifacts["video"], "audio")
    if video_span is not None and audio_span is not None:
        diff = abs(video_span[1] - audio_span[1])
        late = max(0.0, audio_span[0] - video_span[0])
        checks.append(Check(
            name="av_duration", passed=diff <= v.max_av_duration_diff_s and late <= v.max_av_duration_diff_s,
            value=round(max(diff, late), 3), target=f"<= {v.max_av_duration_diff_s} s",
            detail=(f"video {video_span[0]:.3f}-{video_span[1]:.3f} s, "
                    f"audio {audio_span[0]:.3f}-{audio_span[1]:.3f} s"),
        ))
    else:
        checks.append(Check(name="av_duration", passed=False, target="both streams present", detail="missing stream"))

    music_db = rms_db(stem_m)
    music_active = music_db > v.music_present_dbfs
    checks.append(Check(
        name="music_present", passed=music_active, value=round(music_db, 1), target=f"> {v.music_present_dbfs} dB",
    ))

    peak = float(np.max(np.abs(mix)))
    checks.append(Check(name="no_clipping", passed=peak <= 1.0, value=round(peak, 3), target="peak <= 1.0"))

    integrated = lufs(mix, sr)
    if integrated is None:
        checks.append(Check(name="integrated_lufs", passed=False, target=f"{plan.target_lufs} LUFS", detail="mix silent or too short"))
    else:
        checks.append(Check(
            name="integrated_lufs", passed=abs(integrated - plan.target_lufs) <= v.lufs_tolerance_lu,
            value=round(integrated, 2), target=f"{plan.target_lufs} +/- {v.lufs_tolerance_lu} LU",
        ))
    tp = true_peak_db(mix)
    checks.append(Check(
        name="true_peak", passed=tp <= plan.true_peak_dbtp + 1e-3, value=round(tp, 2),
        target=f"<= {plan.true_peak_dbtp} dBTP", detail="4x oversampled, before AAC encoding",
    ))

    if plan.speech_regions:
        margin = rms_db(region_samples(original_m, plan.speech_regions, sr)) - rms_db(
            region_samples(stem_m, plan.speech_regions, sr)
        )
        target = cfg.mix.min_speech_margin_db
        checks.append(Check(
            name="speech_masking", passed=margin >= target, value=round(margin, 1),
            target=f">= {target} dB", detail="clip audio minus music, inside speech regions",
        ))

    if plan.hit_sync == "none" or plan.hit_time is None:
        skipped.append("hit_offset: no hit sync (" + next(
            (w for w in plan.warnings if "hit sync" in w), "not requested") + ")")
    elif music_active:
        # Re-track the placed music rather than trusting the plan, with the
        # engine the plan used: engines disagree by tens of ms and on tempo octave.
        y22 = librosa.resample(stem_m, orig_sr=sr, target_sr=22050)
        engine = "beat_this" if plan.beats_engine == "beat_this" else "librosa"
        tracked = beats_mod.track(y22, 22050, cfg.beats.model_copy(update={"engine": engine}))
        if tracked.times:
            # Timing is judged against the nearest beat. Bar phase is reported but
            # not judged: re-tracking can pick a different bar phase on ambiguous music.
            nearest = min(tracked.times, key=lambda t: abs(t - plan.hit_time))
            offset = abs(nearest - plan.hit_time)
            on_bar = ""
            if plan.hit_sync == "downbeat" and tracked.downbeats_tracked:
                hit_bar = min((abs(d - nearest) for d in tracked.downbeats), default=1.0) < 0.02
                on_bar = f"; on a re-tracked downbeat: {'yes' if hit_bar else 'no'}"
            checks.append(Check(
                name="hit_offset", passed=offset <= v.max_hit_offset_s, value=round(offset, 3),
                target=f"<= {v.max_hit_offset_s} s",
                detail=f"hit at {plan.hit_time:.2f} s to nearest re-tracked beat ({tracked.engine}), "
                       f"{plan.hit_sync} aligned{on_bar}",
            ))
        else:
            checks.append(Check(
                name="hit_offset", passed=False, target=f"<= {v.max_hit_offset_s} s",
                detail="no beats tracked in the placed music",
            ))
    passed = all(c.passed for c in checks)
    return VerifyReport(
        status="ok" if passed else "verify_failed", passed=passed, checks=checks, skipped=skipped,
        integrated_lufs=None if integrated is None else round(integrated, 2), true_peak_dbtp=round(tp, 2),
    )
