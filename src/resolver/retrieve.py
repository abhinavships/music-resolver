"""Brief to three candidates (7.3): multi-query search, fusion, penalties, no-match, slots, MMR, reasons.

Works on a CLAP index (segment embeddings) or, without CLAP, on a tag index
with BM25 over tags and titles (D3). Licence filtering already happened at
ingest, so nothing here can surface an excluded track.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .embed import Embedder, l2
from .ingest import Index
from .library import attribution, licence_warnings
from .schemas import Brief, Candidate, ClipAnalysis, CueSpec, TrackRecord


@dataclass
class Scorer:
    """Similarity of free text (or an audio vector) to every index segment."""
    index: Index
    embedder: Embedder | None
    _bm25: "BM25 | None" = None

    def text(self, texts: list[str]) -> np.ndarray:
        if self.index.tag_mode or self.embedder is None:
            if self._bm25 is None:
                self._bm25 = BM25([_doc(self.index.tracks[t]) for t in self.index.seg_tracks])
            return np.stack([self._bm25.scores(_tokens(t)) for t in texts]) if texts else np.zeros((0, 0))
        return self.embedder.text(texts) @ self.index.vectors.T

    def vector(self, v: np.ndarray) -> np.ndarray:
        if self.index.vectors is None:
            raise ValueError("audio search needs a CLAP index")
        return self.index.vectors @ l2(v)

    def diversity_vectors(self) -> np.ndarray:
        if self.index.vectors is not None:
            return self.index.vectors
        vocab = sorted({w for t in self.index.seg_tracks for w in _doc(self.index.tracks[t])})
        pos = {w: i for i, w in enumerate(vocab)}
        out = np.zeros((len(self.index.seg_tracks), max(1, len(vocab))), dtype=np.float32)
        for i, t in enumerate(self.index.seg_tracks):
            for w in _doc(self.index.tracks[t]):
                out[i, pos[w]] = 1.0
        return l2(out)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _doc(track: TrackRecord) -> list[str]:
    return [w for tag in track.tags for w in _tokens(tag)] + _tokens(track.title)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = np.mean([len(d) for d in docs]) if docs else 0.0
        self.df: dict[str, int] = {}
        for d in docs:
            for w in set(d):
                self.df[w] = self.df.get(w, 0) + 1

    def scores(self, query: list[str]) -> np.ndarray:
        n = len(self.docs)
        out = np.zeros(n)
        for w in set(query):
            df = self.df.get(w, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, d in enumerate(self.docs):
                tf = d.count(w)
                if tf:
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return out


def rrf(rankings: list[np.ndarray], n: int, k: int, top_k: int, weights: list[float] | None = None) -> np.ndarray:
    """Reciprocal rank fusion of per-query similarity arrays over n segments."""
    fused = np.zeros(n)
    for j, sims in enumerate(rankings):
        order = np.argsort(-sims)[:top_k]
        w = weights[j] if weights else 1.0
        fused[order] += w / (k + 1 + np.arange(order.size))
    return fused


def query_z(sims: np.ndarray, background: np.ndarray) -> float:
    """Best similarity against a random background sample of segments."""
    bg = sims[background]
    sd = float(bg.std())
    return float((sims.max() - bg.mean()) / sd) if sd > 1e-9 else (0.0 if sims.max() <= bg.mean() + 1e-9 else 10.0)


def tempo_penalty(bpm: float | None, rng: tuple[int, int] | None, cfg: Config) -> float:
    if bpm is None or rng is None:
        return 0.0
    lo, hi = rng
    # Beat trackers report half or double tempo freely; take the closest reading.
    dist = min(max(0.0, lo - b, b - hi) for b in (bpm, bpm * 2, bpm / 2))
    return min(cfg.retrieve.tempo_penalty_max, cfg.retrieve.tempo_penalty_per_10bpm * dist / 10)


@dataclass
class CueResult:
    cue: CueSpec
    matched: bool
    z: float
    track_scores: dict[str, float]          # after penalties
    track_segment: dict[str, int]           # best segment row per track
    penalties: dict[str, list[str]] = field(default_factory=dict)


def search_cue(cue: CueSpec, scorer: Scorer, clip: ClipAnalysis | None, cfg: Config, exclude: set[str],
               extra_vectors: list[np.ndarray] | None = None, rng: np.random.Generator | None = None) -> CueResult:
    index = scorer.index
    n = len(index.seg_tracks)
    rng = rng or np.random.default_rng(0)
    background = rng.choice(n, size=min(n, cfg.no_match.background_samples), replace=False)
    rankings = [scorer.text([q])[0] for q in cue.queries] if cue.queries else []
    zs = [query_z(s, background) for s in rankings]
    weights = [1.0] * len(rankings)
    for v in extra_vectors or []:
        # Audio-to-audio (R6, R12) leads; the text queries break ties among similar-sounding tracks.
        rankings.append(scorer.vector(v))
        weights.append(cfg.retrieve.audio_query_weight * max(1, len(cue.queries)))
    fused = rrf(rankings, n, cfg.retrieve.rrf_k, cfg.retrieve.per_query_top_k, weights)
    best_possible = sum(w / (cfg.retrieve.rrf_k + 1) for w in weights) or 1.0

    has_dialogue = bool(clip and clip.speech_regions)
    clip_dur = clip.duration if clip else 0.0
    best_seg: dict[str, int] = {}
    for i, t in enumerate(index.seg_tracks):
        if t in exclude:
            continue
        if t not in best_seg or fused[i] > fused[best_seg[t]]:
            best_seg[t] = i
    scores: dict[str, float] = {}
    notes: dict[str, list[str]] = {}
    for t, i in best_seg.items():
        if fused[i] <= 0:
            continue
        track = index.tracks[t]
        s = fused[i] / best_possible
        why: list[str] = []
        if has_dialogue and cue.vocals == "none" and track.vocal_prob:
            p = cfg.retrieve.vocal_penalty * track.vocal_prob
            s -= p
            if track.vocal_prob > 0.5:
                why.append(f"has vocals (p={track.vocal_prob:.2f}) under dialogue")
        tp = tempo_penalty(track.bpm, cue.tempo_bpm, cfg)
        if tp:
            s -= tp
            why.append(f"tempo {track.bpm:.0f} BPM outside {cue.tempo_bpm[0]}-{cue.tempo_bpm[1]}")
        if clip_dur and float(index.seg_starts[i]) + clip_dur > track.duration:
            s -= cfg.retrieve.length_penalty
            why.append("needs a loop to cover the clip")
        scores[t] = s
        notes[t] = why
    z = float(np.median(zs)) if zs else (10.0 if extra_vectors else 0.0)
    matched = bool(extra_vectors) or (z >= cfg.no_match.z_threshold and bool(scores))
    return CueResult(cue, matched, z, scores, best_seg, notes)


def reasons(cue: CueSpec, seg: int, scorer: Scorer) -> list[str]:
    """Brief attributes whose similarity on this segment is in the library's top quartile."""
    attrs = list(dict.fromkeys([*cue.instruments, *cue.mood, f"{cue.energy} energy"]))
    if not attrs:
        return []
    if scorer.index.tag_mode:
        doc = set(_doc(scorer.index.tracks[scorer.index.seg_tracks[seg]]))
        return [a for a in attrs if set(_tokens(a)) & doc]
    sims = scorer.text([f"music with {a}" if a in cue.instruments else f"{a} music" for a in attrs])
    out = []
    for a, row in zip(attrs, sims):
        if row[seg] >= np.percentile(row, 75):
            out.append(a)
    return out


def select(results: list[CueResult], scorer: Scorer, cfg: Config, k: int = 3) -> list[tuple[CueResult, str]]:
    """Allocate slots across cues (R3), one track per artist (B5), MMR within each cue."""
    matched = [r for r in results if r.matched and r.track_scores]
    if not matched:
        return []
    slots = {id(r): (1 if i < k else 0) for i, r in enumerate(matched)}
    vecs = scorer.diversity_vectors()
    chosen: list[tuple[CueResult, str]] = []
    artists: dict[str, int] = {}
    lam = cfg.retrieve.mmr_lambda

    def pick_one(r: CueResult) -> str | None:
        best, best_val = None, -np.inf
        for t, s in r.track_scores.items():
            if any(t == c for _, c in chosen):
                continue
            artist = scorer.index.tracks[t].artist.lower()
            if artists.get(artist, 0) >= cfg.retrieve.max_per_artist:
                continue
            sim = max((float(vecs[r.track_segment[t]] @ vecs[cr.track_segment[c]]) for cr, c in chosen), default=0.0)
            val = lam * s - (1 - lam) * sim
            if val > best_val:
                best, best_val = t, val
        return best

    # One slot per matched cue first (R3), then the rest to the primary; a cue
    # that runs out of tracks hands its slot to the next cue that has one.
    plan = [r for r in matched if slots[id(r)]] + [matched[0]] * (k - min(k, len(matched)))
    for r in plan:
        for cand in [r, *[o for o in matched if o is not r]]:
            t = pick_one(cand)
            if t is not None:
                chosen.append((cand, t))
                artist = scorer.index.tracks[t].artist.lower()
                artists[artist] = artists.get(artist, 0) + 1
                break
    chosen.sort(key=lambda ct: (matched.index(ct[0]), -ct[0].track_scores[ct[1]]))
    return chosen


def coverage_warnings(brief: Brief, index: Index, cfg: Config) -> list[str]:
    inst = index.coverage.get("instruments", {})
    out = []
    for cue in brief.cues():
        for name in cue.instruments:
            key = name.lower()
            if key in inst and inst[key]["segments"] < cfg.retrieve.coverage_min_segments:
                out.append(f"few '{key}' segments in the library ({inst[key]['segments']} < "
                           f"{cfg.retrieve.coverage_min_segments}); matches may be weak (B4)")
    return list(dict.fromkeys(out))


def neighbours(track_ids: list[str], scorer: Scorer, n: int) -> set[str]:
    """The n tracks nearest to any of `track_ids` (R13)."""
    index = scorer.index
    vecs = scorer.diversity_vectors()
    rows = [i for i, t in enumerate(index.seg_tracks) if t in set(track_ids)]
    if not rows:
        return set()
    sims = (vecs @ vecs[rows].T).max(axis=1)
    out: list[str] = []
    for i in np.argsort(-sims):
        t = index.seg_tracks[i]
        if t not in track_ids and t not in out:
            out.append(t)
        if len(out) >= n:
            break
    return set(out)


def retrieve(brief: Brief, index: Index, embedder: Embedder | None, clip: ClipAnalysis | None, cfg: Config,
             exclude: set[str] | None = None, extra_vectors: list[np.ndarray] | None = None,
             k: int = 3) -> tuple[str, list[Candidate], dict, list[str]]:
    """Returns (status, candidates, searched, coverage warnings)."""
    exclude = set(exclude or ())
    scorer = Scorer(index, embedder)
    rng = np.random.default_rng(0)
    results = [search_cue(c, scorer, clip, cfg, exclude, extra_vectors if i == 0 else None, rng)
               for i, c in enumerate(brief.cues())]
    picks = select(results, scorer, cfg, k)
    candidates: list[Candidate] = []
    for rank, (r, t) in enumerate(picks, start=1):
        track = index.tracks[t]
        seg = r.track_segment[t]
        why = reasons(r.cue, seg, scorer)
        warnings = [*licence_warnings(track.licence), *r.penalties.get(t, [])]
        candidates.append(Candidate(
            rank=rank, track=track, segment_start=float(index.seg_starts[seg]), cue_label=r.cue.label,
            score=round(r.track_scores[t], 4), z=round(r.z, 2),
            reason=("matches: " + ", ".join(why)) if why else "closest overall match to the brief",
            warnings=warnings, attribution=attribution(track),
        ))
    searched = {
        "queries": {r.cue.label: r.cue.queries for r in results},
        "z": {r.cue.label: round(r.z, 2) for r in results},
        "matched": {r.cue.label: r.matched for r in results},
        "library_tracks": index.size, "library_segments": len(index.seg_tracks),
        "mode": "tags (BM25, D3)" if index.tag_mode else f"embedding ({index.model_id})",
        "excluded": len(exclude), "profile": cfg.library.profile,
        "audio_query": bool(extra_vectors),
    }
    status = "ok" if candidates else "no_match"
    return status, candidates, searched, coverage_warnings(brief, index, cfg)
