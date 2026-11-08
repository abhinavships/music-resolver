"""Stage 2: find speech regions and rank candidate hit points in a clip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
from scipy.signal import find_peaks

from . import media
from .schemas import ClipAnalysis, HitPoint, Region

SR = 22050
HOP = 512


def speech_regions(y: np.ndarray, sr: int = SR, min_len: float = 0.2, max_gap: float = 0.3) -> list[Region]:
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

    regions: list[list[float]] = []
    for t, on in zip(times, active):
        if not on:
            continue
        if regions and t - regions[-1][1] <= max_gap:
            regions[-1][1] = float(t)
        else:
            regions.append([float(t), float(t)])
    return [Region(start=a, end=b) for a, b in regions if b - a >= min_len]


def audio_onset_curve(y: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=HOP)
    return times, env


def motion_curve(video: str | Path, width: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Local motion per frame. Median flow is subtracted to ignore handheld shake."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev = None
    values: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
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
    # Acceleration marks a hit better than sustained movement.
    change = np.abs(np.diff(arr, prepend=arr[0] if arr.size else 0.0))
    return np.arange(len(change)) / fps, change


def _normalise(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    # Clip to a high percentile so one outlier cannot flatten the rest.
    top = np.percentile(x, 99.5)
    return np.clip(x / top, 0.0, 1.0) if top > 0 else np.zeros_like(x)


def rank_hits(
    audio_t: np.ndarray, audio_env: np.ndarray,
    motion_t: np.ndarray, motion_env: np.ndarray,
    duration: float, top_k: int = 5,
) -> list[HitPoint]:
    grid = np.arange(0.0, duration, 0.02)
    if grid.size == 0:
        return []
    a = np.interp(grid, audio_t, _normalise(audio_env)) if audio_t.size else np.zeros_like(grid)
    m = np.interp(grid, motion_t, _normalise(motion_env)) if motion_t.size else np.zeros_like(grid)
    score = 0.5 * a + 0.5 * m
    peaks, _ = find_peaks(score, distance=25)  # at least 0.5 s apart
    ordered = sorted(peaks, key=lambda i: score[i], reverse=True)[:top_k]
    return [
        HitPoint(
            time=round(float(grid[i]), 3), score=round(float(score[i]), 3),
            audio_score=round(float(a[i]), 3), motion_score=round(float(m[i]), 3),
        )
        for i in ordered
    ]


def analyse(video: str | Path) -> ClipAnalysis:
    duration = media.duration_of(video)
    audio = media.has_audio(video)
    with tempfile.TemporaryDirectory() as tmp:
        y = media.load_audio(video, SR, Path(tmp)) if audio else np.zeros(0, dtype=np.float32)
    regions = speech_regions(y) if y.size else []
    audio_t, audio_env = audio_onset_curve(y) if y.size else (np.zeros(0), np.zeros(0))
    motion_t, motion_env = motion_curve(video)
    return ClipAnalysis(
        path=str(video), duration=duration, has_audio=audio, speech_regions=regions,
        hits=rank_hits(audio_t, audio_env, motion_t, motion_env, duration),
    )
