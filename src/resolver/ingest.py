"""Build the search index: licence filter, decode, segment, embed, tag, coverage (7.2).

Files in `index.dir`:
  manifest.json   model id (or "tags" when CLAP is unavailable, D3), dims, counts
  tracks.jsonl    one TrackRecord per line
  segments.jsonl  (track_id, start) per row of segments.npy
  segments.npy    float16 L2-normalised segment embeddings (CLAP mode only)
  coverage.json   counts per licence, tag and target instrument; exclusions
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from . import media
from .config import Config
from .embed import Embedder, l2
from .library import FilterStats, Stub, filter_stubs
from .mixing import lufs
from .schemas import TrackRecord

TAG_MODE = "tags"
VOCAL_PROMPTS = ("singing voice, vocals, a person singing", "instrumental music, no vocals")


def instrument_prompt(name: str) -> str:
    return f"music with {name}"


@dataclass
class Index:
    dir: Path
    model_id: str
    tracks: dict[str, TrackRecord]
    seg_tracks: list[str]
    seg_starts: np.ndarray
    vectors: np.ndarray | None          # (segments, dim) float32, None in tag mode
    coverage: dict = field(default_factory=dict)

    @property
    def tag_mode(self) -> bool:
        return self.vectors is None

    @property
    def size(self) -> int:
        return len(self.tracks)

    def segments_of(self, track_id: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.seg_tracks) == track_id)


def load_index(index_dir: str | Path) -> Index:
    d = Path(index_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    tracks = {}
    for line in (d / "tracks.jsonl").read_text().splitlines():
        if line.strip():
            t = TrackRecord.model_validate_json(line)
            tracks[t.track_id] = t
    seg_tracks, starts = [], []
    for line in (d / "segments.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            seg_tracks.append(row["track_id"])
            starts.append(row["start"])
    vec_path = d / "segments.npy"
    vectors = np.load(vec_path).astype(np.float32) if manifest["model_id"] != TAG_MODE and vec_path.exists() else None
    coverage = json.loads((d / "coverage.json").read_text()) if (d / "coverage.json").exists() else {}
    return Index(d, manifest["model_id"], tracks, seg_tracks, np.asarray(starts, dtype=float), vectors, coverage)


def segment_starts(duration: float, segment_s: float, hop_s: float, max_s: float) -> list[float]:
    usable = min(duration, max_s)
    if usable <= segment_s:
        return [0.0]
    n = int(np.floor((usable - segment_s) / hop_s)) + 1
    return [round(k * hop_s, 3) for k in range(n)]


def _dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0, 1e-10))


def _bpm(y: np.ndarray, sr: int) -> float | None:
    y22 = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
    try:
        tempo, beats = librosa.beat.beat_track(y=y22[: 22050 * 120], sr=22050)
    except Exception:  # noqa: BLE001
        return None
    tempo = float(np.atleast_1d(tempo)[0])
    return round(tempo, 1) if tempo > 0 and len(beats) >= 4 else None


def _tag_vocal_prob(tags: list[str]) -> float | None:
    s = set(tags)
    if s & {"vocal", "vocals", "singing", "voice"}:
        return 1.0
    if "instrumental" in s:
        return 0.0
    return None


def ingest(cfg: Config, adapters, embedder: Embedder | None, log=print) -> Index:
    """Idempotent: tracks already indexed with the same model id are kept, not re-embedded."""
    out = Path(cfg.index.dir)
    out.mkdir(parents=True, exist_ok=True)
    model_id = embedder.model_id if embedder is not None else TAG_MODE

    old: Index | None = None
    if (out / "manifest.json").exists():
        old = load_index(out)
        if old.model_id != model_id:
            log(f"index model changed ({old.model_id} -> {model_id}); rebuilding")
            old = None

    tracks: dict[str, TrackRecord] = {}
    seg_tracks: list[str] = []
    seg_starts: list[float] = []
    vecs: list[np.ndarray] = []
    if old is not None:
        tracks.update(old.tracks)
        seg_tracks.extend(old.seg_tracks)
        seg_starts.extend(old.seg_starts.tolist())
        if old.vectors is not None:
            vecs.append(old.vectors)

    stats = FilterStats()
    corrupt: list[str] = []
    reused = added = 0
    sr = embedder.sr if embedder is not None else 22050
    vocal_text = embedder.text(list(VOCAL_PROMPTS)) if embedder is not None else None

    stubs: list[Stub] = [s for a in adapters for s in a.stubs()]
    with tempfile.TemporaryDirectory() as tmp:
        for stub, licence in filter_stubs(stubs, cfg.library.profile, stats):
            if stub.track_id in tracks:
                reused += 1
                continue
            try:
                y = media.load_audio(stub.path, sr, Path(tmp))
            except Exception:  # noqa: BLE001  (T7: corrupt or undecodable)
                corrupt.append(stub.track_id)
                continue
            if y.size < sr // 2:
                corrupt.append(stub.track_id)
                continue
            duration = len(y) / sr
            starts = segment_starts(duration, cfg.index.segment_s, cfg.index.hop_s, cfg.index.max_track_s)
            seg_len = int(cfg.index.segment_s * sr)
            windows, kept = [], []
            for s in starts:
                w = y[int(s * sr): int(s * sr) + seg_len]
                if _dbfs(w) < cfg.index.min_segment_dbfs:
                    continue
                windows.append(w)
                kept.append(s)
            if not kept:
                corrupt.append(stub.track_id)
                continue

            vocal_prob = _tag_vocal_prob(stub.tags)
            if embedder is not None:
                v = embedder.audio(windows)
                vecs.append(v.astype(np.float32))
                logits = cfg.embed.zero_shot_scale * (v @ vocal_text.T)
                logits -= logits.max(axis=1, keepdims=True)
                p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
                vocal_prob = round(float(p[:, 0].mean()), 3)
            else:
                kept = [kept[0]]   # tag mode: one entry per track

            seg_tracks.extend([stub.track_id] * len(kept))
            seg_starts.extend(kept)
            level = lufs(y, sr)
            tracks[stub.track_id] = TrackRecord(
                track_id=stub.track_id, title=stub.title, artist=stub.artist, duration=round(duration, 3),
                path=stub.path, source_url=stub.source_url, licence=licence, licence_url=stub.licence_url,
                tags=stub.tags, bpm=_bpm(y, sr), vocal_prob=vocal_prob,
                lufs=None if level is None else round(level, 2),
            )
            added += 1

    # Tracks that left the library (or the profile) are dropped from the index.
    live = {s.track_id for s in stubs} - set(stats.excluded_ids) - set(corrupt)
    vectors = np.concatenate(vecs) if vecs else None
    keep = np.array([t in live for t in seg_tracks], dtype=bool)
    seg_tracks = [t for t, k in zip(seg_tracks, keep) if k]
    seg_starts = [s for s, k in zip(seg_starts, keep) if k]
    if vectors is not None:
        vectors = l2(vectors[keep])
    tracks = {k: v for k, v in tracks.items() if k in live}

    index = Index(out, model_id, tracks, seg_tracks, np.asarray(seg_starts, dtype=float), vectors)
    index.coverage = coverage(index, cfg, embedder, stats, corrupt)
    _write(index)
    log(f"index: {len(tracks)} tracks ({added} added, {reused} already indexed), {len(seg_tracks)} segments, "
        f"excluded {stats.excluded_nd} ND, {stats.excluded_unknown} unknown licence, "
        f"{sum(stats.excluded_profile.values())} by profile, {len(corrupt)} undecodable")
    return index


def coverage(index: Index, cfg: Config, embedder: Embedder | None, stats: FilterStats, corrupt: list[str]) -> dict:
    per_licence: dict[str, dict[str, int]] = {}
    per_tag: dict[str, dict[str, int]] = {}
    seg_count: dict[str, int] = {}
    for t in index.seg_tracks:
        seg_count[t] = seg_count.get(t, 0) + 1
    for tid, track in index.tracks.items():
        n = seg_count.get(tid, 0)
        d = per_licence.setdefault(track.licence, {"tracks": 0, "segments": 0})
        d["tracks"] += 1
        d["segments"] += n
        for tag in track.tags:
            d = per_tag.setdefault(tag, {"tracks": 0, "segments": 0})
            d["tracks"] += 1
            d["segments"] += n

    instruments: dict[str, dict] = {}
    names = cfg.index.coverage_instruments
    if embedder is not None and index.vectors is not None and len(index.seg_tracks):
        sims = index.vectors @ embedder.text([instrument_prompt(n) for n in names]).T
        for j, name in enumerate(names):
            hit = sims[:, j] >= cfg.index.coverage_threshold
            instruments[name] = {
                "segments": int(hit.sum()),
                "tracks": len({t for t, h in zip(index.seg_tracks, hit) if h}),
                "method": "zero-shot",
            }
    else:
        for name in names:
            d = per_tag.get(name, {"tracks": 0, "segments": 0})
            instruments[name] = {**d, "method": "tags"}

    return {
        "model_id": index.model_id,
        "tracks": len(index.tracks),
        "segments": len(index.seg_tracks),
        "profile": cfg.library.profile,
        "per_licence": per_licence,
        "per_tag": dict(sorted(per_tag.items(), key=lambda kv: -kv[1]["tracks"])),
        "instruments": instruments,
        "excluded": {**stats.as_dict(), "undecodable": len(corrupt)},
    }


def _write(index: Index) -> None:
    d = index.dir
    with open(d / "tracks.jsonl", "w") as fh:
        for t in index.tracks.values():
            fh.write(t.model_dump_json() + "\n")
    with open(d / "segments.jsonl", "w") as fh:
        for t, s in zip(index.seg_tracks, index.seg_starts):
            fh.write(json.dumps({"track_id": t, "start": float(s)}) + "\n")
    if index.vectors is not None:
        np.save(d / "segments.npy", index.vectors.astype(np.float16))
    elif (d / "segments.npy").exists():
        (d / "segments.npy").unlink()
    (d / "coverage.json").write_text(json.dumps(index.coverage, indent=2))
    (d / "manifest.json").write_text(json.dumps({
        "model_id": index.model_id, "tracks": len(index.tracks), "segments": len(index.seg_tracks),
        "dim": None if index.vectors is None else int(index.vectors.shape[1]),
    }, indent=2))
