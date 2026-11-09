"""Build a session Context from config: index on disk, a matching embedder, the brief model."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .embed import make_embedder
from .ingest import TAG_MODE, load_index
from .session import Context


def make_context(cfg: Config, offline: bool = False, previews: bool = True) -> tuple[Context, list[str]]:
    """Returns the context and notes on anything degraded (D2, D3)."""
    notes: list[str] = []
    index = None
    embedder = None
    if (Path(cfg.index.dir) / "manifest.json").exists():
        index = load_index(cfg.index.dir)
        if index.model_id != TAG_MODE:
            embedder = make_embedder(index.model_id, cfg.embed.batch_size, cfg.library.cache_dir)
            if embedder is None:
                notes.append(f"{index.model_id} unavailable; cannot search the embedding index (D3)")
                index = None
    else:
        notes.append(f"no index at {cfg.index.dir}; run `resolver index` first")
    return Context(cfg=cfg, index=index, embedder=embedder, offline_brief=offline, previews=previews), notes
