"""JSON artifacts written by each pipeline stage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Region(BaseModel):
    start: float
    end: float


class HitPoint(BaseModel):
    time: float
    score: float
    audio_score: float
    motion_score: float


class ClipAnalysis(BaseModel):
    path: str
    duration: float
    has_audio: bool
    speech_regions: list[Region] = Field(default_factory=list)
    speech_method: str = "energy-band"
    hits: list[HitPoint] = Field(default_factory=list)


class MixPlan(BaseModel):
    clip_path: str
    track_path: str
    clip_duration: float
    hit_time: float | None
    beat_time_in_track: float | None
    track_offset: float
    music_delay: float
    music_gain_db: float
    duck_db: float
    speech_regions: list[Region] = Field(default_factory=list)
    fade_in: float
    fade_out: float
    warnings: list[str] = Field(default_factory=list)


class Check(BaseModel):
    name: str
    passed: bool
    value: float | None = None
    target: str
    detail: str = ""


class VerifyReport(BaseModel):
    passed: bool
    checks: list[Check]
