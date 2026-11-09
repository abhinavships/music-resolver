"""A toy library with known tags and an embedder that can actually hear them.

Each vocabulary word owns one sine frequency. A toy track contains the tones of
its tags; `ToyEmbedder.audio` measures energy at every word's frequency, and
`ToyEmbedder.text` counts the vocabulary words in the text. So text-to-audio
cosine is high exactly when the track carries the words the query names,
which lets retrieval be tested end to end without CLAP.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import soundfile as sf

VOCAB = [
    "dhol", "tabla", "harmonium", "brass", "comic", "playful", "fight", "action", "percussion",
    "ambient", "calm", "piano", "orchestral", "synth", "guitar", "sad", "suspense", "romantic",
    "vocals", "instrumental", "sitar", "taiko", "shehnai", "fast", "slow", "dramatic", "epic", "funk",
]
SYNONYMS = {"singing": "vocals", "voice": "vocals", "drums": "percussion", "quirky": "playful",
            "funny": "comic", "comedy": "comic", "energetic": "fast", "tense": "suspense", "strings": "orchestral"}
SR = 16000
FREQS = {w: 220.0 * (1.09 ** k) for k, w in enumerate(VOCAB)}
WIN = 4096


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        w = SYNONYMS.get(w, w)
        if w in FREQS:
            out.append(w)
    return out


class ToyEmbedder:
    model_id = "toy-tones-v1"
    sr = SR

    def __init__(self):
        self.calls = {"text": 0, "audio": 0}

    def text(self, texts: list[str]) -> np.ndarray:
        self.calls["text"] += len(texts)
        # The last dimension is "words the toy cannot hear": audio never has it,
        # so text with no known word is orthogonal to the whole library.
        out = np.full((len(texts), len(VOCAB) + 1), 0.02, dtype=np.float32)
        for i, t in enumerate(texts):
            words = _words(t)
            for w in words:
                out[i, VOCAB.index(w)] += 1.0
            if not words:
                out[i] = 0.0
                out[i, -1] = 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def audio(self, clips: list[np.ndarray]) -> np.ndarray:
        self.calls["audio"] += len(clips)
        out = np.zeros((len(clips), len(VOCAB) + 1), dtype=np.float32)
        for i, c in enumerate(clips):
            c = np.asarray(c, dtype=np.float32)
            frames = [c[s:s + WIN] for s in range(0, max(1, len(c) - WIN + 1), WIN)] or [c]
            spec = np.zeros(WIN // 2 + 1)
            for f in frames:
                f = np.pad(f, (0, WIN - len(f)))
                spec += np.abs(np.fft.rfft(f * np.hanning(WIN)))
            freqs = np.fft.rfftfreq(WIN, 1 / SR)
            for j, w in enumerate(VOCAB):
                band = np.abs(freqs - FREQS[w]) < 6
                out[i, j] = spec[band].sum()
            out[i, :-1] = np.sqrt(out[i, :-1]) + 1e-3 * (out[i, :-1].max() ** 0.5 + 1e-6)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def make_toy_track(path: Path, tags: list[str], duration: float = 30.0, bpm: float = 120.0,
                   sr: int = 48000, seed: int = 0, late_tags: list[str] | None = None,
                   late_from: float | None = None) -> None:
    """Tones for `tags` (plus "instrumental" unless vocals), clicks at `bpm`.
    `late_tags` are added only from `late_from` seconds (T4: the matching part is not the intro)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    y = 0.003 * rng.standard_normal(t.size)
    words = [SYNONYMS.get(w, w) for w in tags]
    if "vocals" not in words:
        words.append("instrumental")
    for w in words:
        if w in FREQS:
            y += 0.06 * np.sin(2 * np.pi * FREQS[w] * t + rng.uniform(0, 6.28))
    if late_tags:
        gate = (t >= (late_from or 0.0)).astype(float)
        for w in late_tags:
            y += 0.12 * gate * np.sin(2 * np.pi * FREQS[SYNONYMS.get(w, w)] * t)
    period = 60.0 / bpm
    click = np.exp(-np.arange(int(0.03 * sr)) / (0.004 * sr)) * rng.standard_normal(int(0.03 * sr))
    for k, beat in enumerate(np.arange(0, duration - 0.05, period)):
        s = int(beat * sr)
        e = min(len(y), s + click.size)
        y[s:e] += (0.5 if k % 4 == 0 else 0.25) * click[: e - s]
    sf.write(path, np.stack([y, y], axis=1).astype(np.float32), sr)


TOY_LIBRARY = [
    # file, title, artist, licence, tags, bpm
    ("dhol_fight.wav", "Dhol Brawl", "Artist A", "cc-by", ["dhol", "tabla", "fight", "comic", "fast", "percussion"], 140),
    ("tabla_comic.wav", "Tabla Tickle", "Artist A", "cc0", ["tabla", "comic", "playful", "harmonium"], 120),
    ("harmonium_play.wav", "Harmonium Hop", "Artist B", "cc-by", ["harmonium", "playful", "comic"], 110),
    ("brass_action.wav", "Brass Charge", "Artist C", "cc-by-sa", ["brass", "action", "epic", "fast"], 150),
    ("ambient_calm.wav", "Calm Drift", "Artist D", "cc0", ["ambient", "calm", "slow", "synth"], 70),
    ("piano_sad.wav", "Rain Piano", "Artist E", "cc-by-nc", ["piano", "sad", "slow"], 80),
    ("suspense_strings.wav", "Door Creak", "Artist F", "cc-by", ["suspense", "orchestral", "dramatic"], 90),
    ("sitar_romance.wav", "Sitar Dusk", "Artist G", "cc0", ["sitar", "romantic", "slow"], 85),
    ("funk_guitar.wav", "Funk Walk", "Artist H", "cc-by", ["funk", "guitar", "playful"], 115),
    ("taiko_epic.wav", "Taiko Rise", "Artist I", "cc-by", ["taiko", "epic", "percussion", "dramatic"], 130),
    ("vocal_pop.wav", "Sing Along", "Artist J", "cc-by", ["vocals", "playful", "synth"], 120),
    ("dhol_party.wav", "Dhol Party", "Artist K", "cc-by", ["dhol", "fast", "percussion", "playful"], 135),
    ("nd_track.wav", "No Derivs", "Artist L", "https://creativecommons.org/licenses/by-nd/4.0/", ["dhol", "comic"], 120),
    ("mystery.wav", "No Licence", "Artist M", "", ["comic", "tabla"], 120),
]


def build_toy_library(folder: Path, duration: float = 30.0, entries=TOY_LIBRARY) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "licences.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "title", "artist", "licence", "licence_url", "source_url", "tags"])
        for i, (file, title, artist, lic, tags, bpm) in enumerate(entries):
            make_toy_track(folder / file, tags, duration=duration, bpm=bpm, seed=i)
            w.writerow([file, title, artist, lic, "", f"https://example.org/{file}", ";".join(tags)])
    return folder
