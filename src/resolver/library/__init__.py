"""Library adapters. `make_adapters(cfg)` builds the ones listed in `library.adapters`."""

from __future__ import annotations

from ..config import Config
from .base import Adapter, FilterStats, Stub, attribution, filter_stubs, licence_warnings, parse_licence  # noqa: F401


def make_adapters(cfg: Config) -> list[Adapter]:
    out: list[Adapter] = []
    for name in cfg.library.adapters:
        if name == "local_folder":
            from .local_folder import LocalFolder
            out.append(LocalFolder(cfg.library.local_folder))
        elif name == "jamendo_api":
            from .jamendo_api import JamendoApi
            out.append(JamendoApi(cfg.library.cache_dir, cfg.library.jamendo_tags, cfg.library.jamendo_limit))
        elif name == "mtg_jamendo":
            from .mtg_jamendo import MtgJamendo
            out.append(MtgJamendo(cfg.library.mtg_jamendo_root, tags=cfg.library.jamendo_tags))
        else:
            raise ValueError(f"unknown adapter {name!r}")
    return out
