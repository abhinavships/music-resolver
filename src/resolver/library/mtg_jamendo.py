"""MTG-Jamendo subset (https://mtg.github.io/mtg-jamendo-dataset).

Reads the dataset's tag TSV (`TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS...`)
plus a `licences.tsv` (`TRACK_ID<TAB>license_ccurl`) built from the Jamendo API,
since the dataset TSVs do not carry per-track licences. A track with no licence
row is excluded (L6). The dataset as a whole is for non-commercial research use.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .base import Stub


class MtgJamendo:
    name = "mtg_jamendo"

    def __init__(self, root: str | Path, tsv: str = "autotagging.tsv", licences: str = "licences.tsv",
                 tags: list[str] | None = None):
        self.root = Path(root)
        self.tsv, self.licences = self.root / tsv, self.root / licences
        self.want = {t.lower() for t in tags} if tags else None

    def _licences(self) -> dict[str, str]:
        if not self.licences.exists():
            return {}
        with open(self.licences, newline="", encoding="utf-8") as fh:
            return {row[0]: row[1] for row in csv.reader(fh, delimiter="\t") if len(row) >= 2}

    def stubs(self) -> Iterator[Stub]:
        licences = self._licences()
        with open(self.tsv, encoding="utf-8") as fh:
            next(fh, None)
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 5:
                    continue
                track_id, artist_id, _album, rel, duration, *raw_tags = cols
                tags = [t.split("---", 1)[-1].lower() for t in raw_tags]
                if self.want and not self.want & set(tags):
                    continue
                num = track_id.removeprefix("track_").lstrip("0")
                yield Stub(
                    track_id=f"{self.name}:{track_id}",
                    title=track_id,
                    artist=artist_id,
                    path=str(self.root / "audio" / rel),
                    licence_raw=licences.get(track_id) or licences.get(num),
                    source_url=f"https://www.jamendo.com/track/{num}" if num else None,
                    tags=tags,
                    duration=float(duration) if duration else None,
                )
