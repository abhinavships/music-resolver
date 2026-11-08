"""Synthetic media with known ground truth."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter

from resolver import media

SR = 48000
FPS = 30
DURATION = 12.0
HIT_TIME = 7.0
SPEECH = [(1.0, 3.0), (4.5, 6.0)]
BPM = 120.0


def _speech_like(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Voice-like signal: a glottal pulse train with a moving pitch, through
    vowel formants, at syllable rate. Speech enough for Silero VAD as well as
    the energy-band detector."""
    t = np.arange(n) / sr
    f0 = 125 + 25 * np.sin(2 * np.pi * 1.3 * t) + 10 * np.sin(2 * np.pi * 3.1 * t)
    pulses = (np.diff(np.floor(np.cumsum(f0 / sr)), prepend=0) > 0).astype(float)
    source = lfilter([1], [1, -0.97], pulses) + 0.02 * rng.standard_normal(n)
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (660, 1720, 2410)]
    syllable = int(0.18 * sr)
    out = np.zeros(n)
    for k, s in enumerate(range(0, n, syllable)):
        e = min(n, s + syllable)
        seg = source[s:e]
        y = np.zeros_like(seg)
        for f, bw in zip(vowels[k % len(vowels)], (90, 110, 170)):
            r = np.exp(-np.pi * bw / sr)
            theta = 2 * np.pi * f / sr
            y += lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], seg)
        out[s:e] = y * (0.35 + 0.65 * np.sin(np.pi * np.arange(e - s) / syllable) ** 0.7)
    return 0.3 * out / np.max(np.abs(out))


def make_clip(path: Path, tmp: Path) -> None:
    rng = np.random.default_rng(0)
    video_path = tmp / "silent.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 180))
    x = 100
    for i in range(int(DURATION * FPS)):
        t = i / FPS
        frame = np.full((180, 320, 3), 40, dtype=np.uint8)
        frame[::20, :] = 90
        frame[:, ::20] = 90
        # A square sits still, then jumps sideways at the hit.
        if abs(t - HIT_TIME) < 1 / FPS:
            x = 220
        cv2.rectangle(frame, (x, 60), (x + 50, 110), (230, 230, 230), -1)
        writer.write(frame)
    writer.release()

    n = int(DURATION * SR)
    audio = 0.002 * rng.standard_normal(n)
    for a, b in SPEECH:
        s, e = int(a * SR), int(b * SR)
        audio[s:e] += _speech_like(e - s, SR, rng)
    # A sharp impact at the hit, below the speech band's main energy.
    s = int(HIT_TIME * SR)
    length = int(0.05 * SR)
    audio[s:s + length] += 0.9 * rng.standard_normal(length) * np.exp(-np.arange(length) / (0.01 * SR))
    wav = tmp / "clip.wav"
    sf.write(wav, audio.astype(np.float32), SR)
    media.mux(video_path, wav, path)


def make_track(path: Path, duration: float = 30.0, bpm: float = BPM, sr: int = SR) -> None:
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    length = int(0.08 * sr)
    click = np.sin(2 * np.pi * 110 * np.arange(length) / sr) * np.exp(-np.arange(length) / (0.02 * sr))
    for k, t in enumerate(np.arange(0.5, duration - 0.2, period)):
        s = int(t * sr)
        y[s:s + length] += (0.9 if k % 4 == 0 else 0.5) * click.astype(np.float32)
    sf.write(path, y, sr)


def make_rich_track(path: Path, duration: float = 30.0, bpm: float = BPM, sr: int = SR) -> None:
    """Kick, syncopated off-beat hits, noisy snare and a sustained pad."""
    rng = np.random.default_rng(int(bpm))
    n = int(duration * sr)
    t_all = np.arange(n) / sr
    y = 0.08 * (np.sin(2 * np.pi * 196 * t_all) + np.sin(2 * np.pi * 294 * t_all))
    period = 60.0 / bpm
    length = int(0.12 * sr)
    t = np.arange(length) / sr
    kick = np.sin(2 * np.pi * 70 * t) * np.exp(-t / 0.03)
    tom = np.sin(2 * np.pi * 240 * t) * np.exp(-t / 0.02)
    snare = rng.standard_normal(length) * np.exp(-t / 0.02)
    for k in range(int((duration - 0.5) / period)):
        s = int((0.3 + k * period) * sr)
        y[s:s + length] += 0.8 * kick
        if k % 2 == 1:
            y[s:s + length] += 0.35 * snare
        off = s + int(0.75 * period * sr)
        if k % 4 in (0, 3) and off + length < n:
            y[off:off + length] += 0.3 * tom
    y *= 0.8 / np.max(np.abs(y))
    sf.write(path, y.astype(np.float32), sr)


@pytest.fixture(scope="session")
def clip_path(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("clip")
    path = tmp / "clip.mp4"
    make_clip(path, tmp)
    return path


@pytest.fixture(scope="session")
def track_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("track") / "track.wav"
    make_track(path)
    return path
