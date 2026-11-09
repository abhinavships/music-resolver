"""Audio files in a folder, described by `licences.csv`.

Columns: file, title, artist, licence, licence_url, source_url, tags (separated by `;`).
A file without a row has no licence and is excluded (L6).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .base import Stub

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


class LocalFolder:
    name = "local_folder"

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def stubs(self) -> Iterator[Stub]:
        rows: dict[str, dict] = {}
        sheet = self.folder / "licences.csv"
        if sheet.exists():
            with open(sheet, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rows[(row.get("file") or "").strip()] = row
        for path in sorted(p for p in self.folder.rglob("*") if p.suffix.lower() in AUDIO):
            rel = path.relative_to(self.folder).as_posix()
            row = rows.get(rel) or rows.get(path.name) or {}
            yield Stub(
                track_id=f"{self.name}:{rel}",
                title=(row.get("title") or path.stem).strip(),
                artist=(row.get("artist") or "unknown").strip(),
                path=str(path),
                licence_raw=(row.get("licence") or "").strip() or None,
                licence_url=(row.get("licence_url") or "").strip() or None,
                source_url=(row.get("source_url") or "").strip() or None,
                tags=[t.strip().lower() for t in (row.get("tags") or "").split(";") if t.strip()],
            )
