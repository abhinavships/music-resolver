"""ffmpeg and ffprobe helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr[-1500:]}")
    return result


def probe(path: str | Path) -> dict:
    out = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]).stdout
    return json.loads(out)


def duration_of(path: str | Path) -> float:
    return float(probe(path)["format"]["duration"])


def stream_duration(path: str | Path, kind: str) -> float | None:
    for stream in probe(path)["streams"]:
        if stream["codec_type"] == kind and "duration" in stream:
            return float(stream["duration"])
    return None


def has_audio(path: str | Path) -> bool:
    return any(s["codec_type"] == "audio" for s in probe(path)["streams"])


def load_audio(path: str | Path, sr: int, tmp_dir: Path, mono: bool = True) -> np.ndarray:
    """Decode any media file's audio. Returns (samples,) or (samples, channels)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav = tmp_dir / f"decoded_{Path(path).stem}_{sr}_{int(mono)}.wav"
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn",
        "-ac", "1" if mono else "2", "-ar", str(sr), "-c:a", "pcm_f32le", str(wav),
    ])
    audio, _ = sf.read(wav, dtype="float32", always_2d=not mono)
    return audio


def mux(video: str | Path, audio_wav: str | Path, out: str | Path) -> None:
    """Copy the video stream untouched and attach the new audio."""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio_wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(out),
    ])
