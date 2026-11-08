"""Stage 2: find speech regions and rank candidate hit points in a clip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
from scipy.signal import find_peaks

from . import media, vad
from .config import Config, HitsConfig
from .schemas import ClipAnalysis, HitPoint, Region

SR = 22050
HOP = 512


def speech_regions(y: np.ndarray, sr: int = SR) -> list[Region]:
    """Energy-band detector, kept for direct use and tests. See `vad.speech_regions`."""
    return vad.energy_regions(y, sr)


def audio_onset_curve(y: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=HOP)
    return times, env


def motion_curve(video: str | Path, width: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Local motion change per frame, timed by each frame's own timestamp.

    Phone and WhatsApp video often has a variable frame rate, so frame index
    divided by the nominal rate drifts (C5). Times count from the video
    stream's start; `analyse` moves them onto the clip timeline.
    Median flow is subtracted to ignore handheld shake.
    """
    cap = cv2.VideoCapture(str(video))
    prev = None
    values: list[float] = []
    stamps: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, max(2, int(h * width / w))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is None:
            values.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            flow -= np.median(flow.reshape(-1, 2), axis=0)
            mag = np.linalg.norm(flow, axis=2)
            values.append(float(np.percentile(mag, 95)))
        prev = gray
    cap.release()
    arr = np.asarray(values, dtype=float)
    times = np.asarray(stamps, dtype=float)
    if times.size > 1 and not np.all(np.diff(times) > 0):
        # Some containers report no timestamps; fall back to the nominal rate.
        fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
        times = np.arange(len(arr)) / fps
    # Acceleration marks a hit better than sustained movement.
    change = np.abs(np.diff(arr, prepend=arr[0] if arr.size else 0.0))
    return times, change


def _normalise(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    # Clip to a high percentile so one outlier cannot flatten the rest.
    top = np.percentile(x, 99.5)
    return np.clip(x / top, 0.0, 1.0) if top > 0 else np.zeros_like(x)


def salience(x: np.ndarray, cfg: HitsConfig, floor: float = 0.0) -> float:
    """0..1: how far the curve's peak stands above its typical level.

    `_normalise` is relative, so on a clip with no event the largest wobble
    still scores 1. Salience scales it down so C7 can detect "no clear hit".
    """
    if x.size == 0:
        return 0.0
    peak = float(np.max(x))
    if peak <= floor:
        return 0.0
    median = float(np.median(x))
    spread = 1.4826 * float(np.median(np.abs(x - median)))
    spread = max(spread, floor / cfg.salience_z_low, 1e-9)
    z = (peak - median) / spread
    return float(np.clip((z - cfg.salience_z_low) / (cfg.salience_z_high - cfg.salience_z_low), 0.0, 1.0))


def rank_hits(
    audio_t: np.ndarray, audio_env: np.ndarray,
    motion_t: np.ndarray, motion_env: np.ndarray,
    duration: float, top_k: int | None = None, cfg: HitsConfig = HitsConfig(),
) -> list[HitPoint]:
    top_k = top_k or cfg.top_k
    step = 0.02
    grid = np.arange(0.0, duration, step)
    if grid.size == 0:
        return []
    sal_a = salience(audio_env, cfg)
    sal_m = salience(motion_env, cfg, floor=cfg.motion_floor_px)
    a = np.interp(grid, audio_t, _normalise(audio_env), left=0.0, right=0.0) if audio_t.size else np.zeros_like(grid)
    m = np.interp(grid, motion_t, _normalise(motion_env), left=0.0, right=0.0) if motion_t.size else np.zeros_like(grid)
    a, m = a * sal_a, m * sal_m
    score = 0.5 * a + 0.5 * m
    peaks, _ = find_peaks(score, distance=max(1, int(cfg.min_spacing_s / step)))
    ordered = sorted(peaks, key=lambda i: score[i], reverse=True)[:top_k]
    return [
        HitPoint(
            time=round(float(grid[i]), 3), score=round(float(score[i]), 3),
            audio_score=round(float(a[i]), 3), motion_score=round(float(m[i]), 3),
        )
        for i in ordered
    ]


def analyse(video: str | Path, cfg: Config | None = None) -> ClipAnalysis:
    cfg = cfg or Config()
    duration = media.duration_of(video)
    audio = media.has_audio(video)
    audio_offset = media.stream_offset(video, "audio") if audio else 0.0
    video_offset = media.stream_offset(video, "video")
    with tempfile.TemporaryDirectory() as tmp:
        y = media.load_audio(video, SR, Path(tmp)) if audio else np.zeros(0, dtype=np.float32)

    regions, method = vad.speech_regions(y, SR, cfg.vad) if y.size else ([], "none")
    regions = [Region(start=round(r.start + audio_offset, 3), end=round(r.end + audio_offset, 3)) for r in regions]
    audio_t, audio_env = audio_onset_curve(y) if y.size else (np.zeros(0), np.zeros(0))
    motion_t, motion_env = motion_curve(video, cfg.hits.motion_width_px)
    hits = rank_hits(
        audio_t + audio_offset, audio_env, motion_t + video_offset, motion_env, duration, cfg=cfg.hits,
    )
    return ClipAnalysis(
        path=str(video), duration=duration, has_audio=audio, speech_regions=regions,
        speech_method=method, hits=hits, audio_offset=round(audio_offset, 4),
        video_offset=round(video_offset, 4),
    )
