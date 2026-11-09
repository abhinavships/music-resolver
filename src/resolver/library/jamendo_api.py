"""Jamendo API v3 adapter (https://developer.jamendo.com/v3.0/tracks).

Needs a free `JAMENDO_CLIENT_ID`. Only tracks with `audiodownload_allowed` are
used; audio is cached under `<cache_dir>/jamendo/`. The licence comes from
`license_ccurl` and is filtered by the caller like any other adapter.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from .base import Stub

API = "https://api.jamendo.com/v3.0/tracks/"
PAGE = 200

Fetch = Callable[[str], dict]


def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "music-resolver/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "music-resolver/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    tmp.rename(dest)


def track_tags(item: dict) -> list[str]:
    info = (item.get("musicinfo") or {}).get("tags") or {}
    tags = [*info.get("genres", []), *info.get("instruments", []), *info.get("vartags", [])]
    vi = (item.get("musicinfo") or {}).get("vocalinstrumental")
    if vi:
        tags.append(vi)
    return sorted({t.lower() for t in tags if t})


class JamendoApi:
    name = "jamendo_api"

    def __init__(
        self, cache_dir: str | Path, tags: list[str], limit: int = 2000, client_id: str | None = None,
        fetch: Fetch = http_json, download: Callable[[str, Path], None] = http_download,
    ):
        self.client_id = client_id or os.environ.get("JAMENDO_CLIENT_ID")
        self.cache = Path(cache_dir) / "jamendo"
        self.tags, self.limit = tags, limit
        self.fetch, self.download = fetch, download

    def page_url(self, tag: str, offset: int) -> str:
        query = {
            "client_id": self.client_id, "format": "json", "limit": PAGE, "offset": offset,
            "tags": tag, "include": "musicinfo licenses", "audiodlformat": "mp32", "order": "popularity_total",
        }
        return API + "?" + urllib.parse.urlencode(query)

    def items(self) -> Iterator[dict]:
        if not self.client_id:
            raise RuntimeError("JAMENDO_CLIENT_ID is not set (free id from https://devportal.jamendo.com)")
        seen: set[str] = set()
        per_tag = max(1, self.limit // max(1, len(self.tags)))
        for tag in self.tags:
            offset = 0
            while offset < per_tag:
                data = self.fetch(self.page_url(tag, offset))
                results = data.get("results", [])
                for item in results:
                    if str(item["id"]) not in seen:
                        seen.add(str(item["id"]))
                        yield item
                if len(results) < PAGE:
                    break
                offset += PAGE

    def stubs(self) -> Iterator[Stub]:
        for item in self.items():
            if not item.get("audiodownload_allowed", False) or not item.get("audiodownload"):
                continue
            dest = self.cache / f"{item['id']}.mp3"
            if not dest.exists():
                try:
                    self.download(item["audiodownload"], dest)
                except Exception:  # noqa: BLE001  (network errors: skip the track, keep going)
                    continue
            yield Stub(
                track_id=f"{self.name}:{item['id']}",
                title=item.get("name") or str(item["id"]),
                artist=item.get("artist_name") or "unknown",
                path=str(dest),
                licence_raw=item.get("license_ccurl"),
                licence_url=item.get("license_ccurl"),
                source_url=item.get("shareurl"),
                tags=track_tags(item),
                duration=float(item["duration"]) if item.get("duration") else None,
            )
