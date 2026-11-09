"""Optional effects layer (R9, F7): one CC0 effect per secondary hit, peak-aligned.

Effects come from a local folder with a `licences.csv` (CC0 rows only), which
the Freesound helper below can fill. Off unless `sfx.enabled`.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from . import media
from .config import Config
from .library import LocalFolder, parse_licence
from .library.base import LicenceError
from .schemas import ClipAnalysis


def cc0_effects(folder: str | Path) -> list[Path]:
    out = []
    for stub in LocalFolder(folder).stubs():
        try:
            if parse_licence(stub.licence_raw or stub.licence_url) == "cc0":
                out.append(Path(stub.path))
        except LicenceError:
            continue
    return out


def plan_sfx(clip: ClipAnalysis, cfg: Config, effects: list[Path] | None = None) -> list[dict]:
    if not cfg.sfx.enabled:
        return []
    effects = effects if effects is not None else cc0_effects(cfg.sfx.folder)
    if not effects:
        return []
    secondary = [h for h in clip.hits[1:] if h.score >= cfg.sfx.min_hit_score][: cfg.sfx.max_effects]
    return [
        {"path": str(effects[i % len(effects)]), "time": h.time, "gain_db": cfg.sfx.gain_db, "licence": "cc0"}
        for i, h in enumerate(sorted(secondary, key=lambda h: h.time))
    ]


def render_sfx(sfx: list[dict], n: int, sr: int, work: Path) -> np.ndarray:
    """Stem with each effect's peak on its hit, peak level `gain_db` dBFS."""
    stem = np.zeros((n, 2), dtype=np.float32)
    for item in sfx:
        y = media.load_audio(item["path"], sr, work, mono=False)
        mono = np.abs(y).max(axis=1)
        if not mono.size or mono.max() <= 0:
            continue
        y = y / mono.max() * 10 ** (item["gain_db"] / 20)
        peak = int(np.argmax(mono))
        start = int(round(item["time"] * sr)) - peak
        a, b = max(0, start), min(n, start + len(y))
        if b > a:
            stem[a:b] += y[a - start:b - start]
    return stem


def freesound_fetch(query: str, dest: Path, n: int = 5, token: str | None = None) -> list[Path]:
    """Download up to n CC0 effects for `query` into `dest` and record them in licences.csv (live only)."""
    token = token or os.environ.get("FREESOUND_API_KEY")
    if not token:
        raise RuntimeError("FREESOUND_API_KEY is not set")
    q = urllib.parse.urlencode({"query": query, "filter": 'license:"Creative Commons 0"',
                                "fields": "id,name,username,license,previews,url", "page_size": n, "token": token})
    with urllib.request.urlopen(f"https://freesound.org/apiv2/search/text/?{q}", timeout=30) as resp:
        results = json.loads(resp.read())["results"]
    dest.mkdir(parents=True, exist_ok=True)
    sheet = dest / "licences.csv"
    new = not sheet.exists()
    out = []
    with open(sheet, "a") as fh:
        if new:
            fh.write("file,title,artist,licence,licence_url,source_url,tags\n")
        for r in results:
            if "zero" not in r["license"] and "publicdomain" not in r["license"]:
                continue
            path = dest / f"fs_{r['id']}.mp3"
            urllib.request.urlretrieve(r["previews"]["preview-hq-mp3"], path)
            fh.write(f"{path.name},{r['name'].replace(',', ' ')},{r['username']},cc0,{r['license']},{r['url']},{query}\n")
            out.append(path)
    return out
