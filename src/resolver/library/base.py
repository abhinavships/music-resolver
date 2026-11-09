"""Track stubs from adapters, licence parsing and licence filtering (L1 to L7).

Filtering happens here, before anything is decoded, embedded or ranked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Protocol

from ..schemas import TrackRecord

# Licences each profile may show. ND is never here (L5); unknown never parses (L6).
PROFILES: dict[str, set[str]] = {
    "commercial": {"cc0", "pd", "cc-by", "own"},
    "demo": {"cc0", "pd", "cc-by", "own", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"},
}

WARNINGS = {
    "cc-by-sa": "CC BY-SA: the finished edit may have to carry the same licence (L3).",
    "cc-by-nc": "CC BY-NC: non-commercial use only (L4).",
    "cc-by-nc-sa": "CC BY-NC-SA: non-commercial use only, and the edit may have to carry the same licence (L3, L4).",
}

LICENCE_URLS = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "pd": "https://creativecommons.org/publicdomain/mark/1.0/",
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
}


class LicenceError(ValueError):
    """The licence is No-Derivatives (L5) or cannot be read (L6)."""

    def __init__(self, code: str, raw: str):
        super().__init__(f"{code}: {raw!r}")
        self.code = code  # "nd" or "unknown"
        self.raw = raw


def parse_licence(raw: str | None) -> str:
    """Normalise a licence URL or name ("http://creativecommons.org/licenses/by-nc-sa/3.0/",
    "CC BY 4.0", "cc0") to a `Licence` code. Raises LicenceError for ND or unknown."""
    if raw is None or not str(raw).strip():
        raise LicenceError("unknown", str(raw))
    text = str(raw).strip().lower()
    if "publicdomain/zero" in text or re.fullmatch(r"(cc[\s_-]?0|cc[\s_-]?zero)(\s*1\.0)?(\s+universal)?", text):
        return "cc0"
    if "publicdomain/mark" in text or text in {"pd", "public domain", "public-domain", "pdm"}:
        return "pd"
    if text in {"own", "own file"}:
        return "own"
    url = re.search(r"creativecommons\.org/licenses/([a-z-]+)", text)
    if url:
        parts = url.group(1).split("-")
    else:
        m = re.fullmatch(r"(?:cc[\s_-]+)?(by(?:[\s_-]+(?:nc|nd|sa))*)(?:[\s_-]+\d\.\d)?(?:[\s_-]+\w+)?", text)
        if not m:
            raise LicenceError("unknown", str(raw))
        parts = re.split(r"[\s_-]+", m.group(1))
    if not parts or parts[0] != "by":
        raise LicenceError("unknown", str(raw))
    elements = set(parts[1:])
    if "nd" in elements:
        raise LicenceError("nd", str(raw))
    if not elements <= {"nc", "sa"}:
        raise LicenceError("unknown", str(raw))
    code = "cc-by" + ("-nc" if "nc" in elements else "") + ("-sa" if "sa" in elements else "")
    return code


def allowed(licence: str, profile: str) -> bool:
    return licence in PROFILES[profile]


def licence_warnings(licence: str) -> list[str]:
    return [WARNINGS[licence]] if licence in WARNINGS else []


def attribution(track: TrackRecord) -> str | None:
    """Ready-to-paste credit line (L2). Required for every CC BY variant."""
    if not track.licence.startswith("cc-by"):
        return None
    name = track.licence.upper().replace("CC-", "CC ").replace("-", "-")
    url = track.licence_url or LICENCE_URLS.get(track.licence, "")
    parts = [f'"{track.title}" by {track.artist}', f"licensed under {name}"]
    if url:
        parts[-1] += f" ({url})"
    if track.source_url:
        parts.append(f"source: {track.source_url}")
    return ", ".join(parts)


@dataclass
class Stub:
    """What an adapter knows before ingest: metadata plus a raw licence string."""
    track_id: str
    title: str
    artist: str
    path: str
    licence_raw: str | None
    source_url: str | None = None
    licence_url: str | None = None
    tags: list[str] = field(default_factory=list)
    duration: float | None = None


class Adapter(Protocol):
    name: str

    def stubs(self) -> Iterable[Stub]: ...


@dataclass
class FilterStats:
    kept: int = 0
    excluded_nd: int = 0
    excluded_unknown: int = 0
    excluded_profile: dict[str, int] = field(default_factory=dict)
    excluded_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kept": self.kept, "excluded_nd": self.excluded_nd, "excluded_unknown": self.excluded_unknown,
            "excluded_profile": self.excluded_profile,
        }


def filter_stubs(stubs: Iterable[Stub], profile: str, stats: FilterStats) -> Iterator[tuple[Stub, str]]:
    """Yield (stub, licence) for stubs the profile allows; count the rest."""
    for stub in stubs:
        try:
            code = parse_licence(stub.licence_raw or stub.licence_url)
        except LicenceError as err:
            if err.code == "nd":
                stats.excluded_nd += 1
            else:
                stats.excluded_unknown += 1
            stats.excluded_ids.append(stub.track_id)
            continue
        if not allowed(code, profile):
            stats.excluded_profile[code] = stats.excluded_profile.get(code, 0) + 1
            stats.excluded_ids.append(stub.track_id)
            continue
        stats.kept += 1
        yield stub, code
