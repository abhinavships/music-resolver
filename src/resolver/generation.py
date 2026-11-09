"""Music generation: interface only in v1 (`generation.provider = "none"`).

A provider quotes a price first; nothing is generated, or charged, unless the
caller passes back exactly the quoted cost (the fallback enforces this too).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import Config
from .schemas import Brief


class GenerationProvider(Protocol):
    name: str

    def quote(self, brief: Brief, duration: float) -> float: ...
    def generate(self, brief: Brief, duration: float, out: Path) -> Path: ...


PROVIDERS: dict[str, type] = {}


def make_provider(cfg: Config) -> GenerationProvider | None:
    if cfg.generation.provider == "none":
        return None
    try:
        return PROVIDERS[cfg.generation.provider]()
    except KeyError as exc:
        raise ValueError(f"unknown generation provider {cfg.generation.provider!r}") from exc
