"""Stage 4 render: mix music under the clip's own audio and mux with the original video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from . import media, mixing, segment, sfx as sfx_mod
from .config import Config
from .schemas import MixPlan

SR = 48000


def place_music(plan: MixPlan, track: np.ndarray, n: int, sr: int = SR) -> np.ndarray:
    """Put the (looped, if planned) track on the clip timeline with fades. No level changes."""
    if plan.loop is not None:
        track = segment.build_looped(track, sr, plan.loop)
    stem = np.zeros((n, 2), dtype=np.float32)
    start = int(round(plan.music_delay * sr))
    begin = int(round(plan.track_offset * sr))
    length = min(n - start, len(track) - begin)
    if length > 0:
        stem[start:start + length] = track[begin:begin + length]

    fade_out = int(plan.fade_out * sr)
    if fade_out:
        stem[n - fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    fade_in = int(plan.fade_in * sr)
    if fade_in and length > 0:
        stem[start:start + fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    return stem


def load_original(clip_path: str, n: int, work: Path, sr: int = SR) -> np.ndarray:
    """Clip audio on the clip timeline: padded by the audio stream's start offset (C6)."""
    out = np.zeros((n, 2), dtype=np.float32)
    if not media.has_audio(clip_path):
        return out
    audio = media.load_audio(clip_path, sr, work, mono=False)
    lead = int(round(media.stream_offset(clip_path, "audio") * sr))
    if lead < n:
        take = min(n - lead, len(audio))
        out[lead:lead + take] = audio[:take]
    return out


def render(
    plan: MixPlan, out: str | Path, work_dir: str | Path, cfg: Config | None = None,
) -> tuple[MixPlan, dict[str, str]]:
    """Returns the plan with resolved levels, and the artifact paths."""
    cfg = cfg or Config()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    n = int(plan.clip_duration * SR)

    original = load_original(plan.clip_path, n, work)
    track = media.load_audio(plan.track_path, SR, work, mono=False)
    music = place_music(plan, track, n)
    extra = sfx_mod.render_sfx(plan.sfx, n, SR, work) if plan.sfx else None
    original, stem, mix, resolved = mixing.mix(plan, original, music, SR, cfg.mix, extra=extra)

    paths = {
        "original": work / "original.wav",
        "music_stem": work / "music_stem.wav",
        "mix": work / "mix.wav",
    }
    sf.write(paths["original"], original.astype(np.float32), SR, subtype="FLOAT")
    sf.write(paths["music_stem"], stem.astype(np.float32), SR, subtype="FLOAT")
    sf.write(paths["mix"], mix.astype(np.float32), SR, subtype="FLOAT")
    media.mux(plan.clip_path, paths["mix"], out)
    return resolved, {**{k: str(v) for k, v in paths.items()}, "video": str(out)}


def preview(video: str | Path, out: str | Path, height: int = 360) -> None:
    """Small, fast encode of a rendered candidate for the review page."""
    media._run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart", str(out),
    ])
