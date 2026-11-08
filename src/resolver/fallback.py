"""Option order and skip handling.

A skip or empty result always moves to the next option. Nothing is charged
without explicit approval, and the last option always explains what was tried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Option(str, Enum):
    LICENSED = "licensed"
    OWN_FILE = "own_file"
    GENERATE = "generate"
    NONE = "none"


ORDER = [Option.LICENSED, Option.OWN_FILE, Option.GENERATE, Option.NONE]


class ChargeNotApproved(Exception):
    """Raised when a paid option is picked without approval of its stated cost."""


@dataclass
class Fallback:
    generation_cost: float | None = None
    current: Option = Option.LICENSED
    tried: list[str] = field(default_factory=list)
    chosen: Option | None = None
    charged: float = 0.0

    def __post_init__(self) -> None:
        self._skip_unavailable()

    @property
    def finished(self) -> bool:
        return self.chosen is not None

    def _skip_unavailable(self) -> None:
        if self.current is Option.GENERATE and self.generation_cost is None:
            self.tried.append("generate: no provider configured")
            self.current = Option.NONE
        if self.current is Option.NONE:
            self.chosen = Option.NONE

    def _advance(self, note: str) -> Option:
        if self.finished:
            raise ValueError("fallback already finished")
        self.tried.append(f"{self.current.value}: {note}")
        self.current = ORDER[ORDER.index(self.current) + 1]
        self._skip_unavailable()
        return self.current

    def skip(self, reason: str = "skipped by user") -> Option:
        return self._advance(reason)

    def no_match(self, searched: str) -> Option:
        return self._advance(f"no match, searched {searched}")

    def pick(self, approved_cost: float | None = None) -> Option:
        if self.finished:
            raise ValueError("fallback already finished")
        if self.current is Option.GENERATE:
            if approved_cost is None or approved_cost != self.generation_cost:
                raise ChargeNotApproved(
                    f"generation costs {self.generation_cost}; approved {approved_cost}"
                )
            self.charged = approved_cost
        self.chosen = self.current
        return self.chosen

    def summary(self) -> str:
        tried = "; ".join(self.tried) or "nothing"
        return f"Result: {self.chosen.value if self.chosen else 'pending'}. Tried: {tried}."
