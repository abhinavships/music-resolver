"""JSON artifacts written by each pipeline stage."""

from __future__ import annotations

from typing import Literal

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
    # Stream start minus container start, already applied to the times above (C6).
    audio_offset: float = 0.0
    video_offset: float = 0.0


class Loop(BaseModel):
    """Whole bars [start, end) of the track repeated to cover the clip (T3)."""
    start: float
    end: float
    repeats: int
    crossfade_ms: float


class MixPlan(BaseModel):
    clip_path: str
    track_path: str
    clip_duration: float
    hit_time: float | None
    hit_sync: Literal["downbeat", "beat", "none"] = "beat"
    downbeat_used: bool = False
    beats_engine: str = "librosa"
    beat_confidence: float | None = None
    beat_time_in_track: float | None
    segment_start: float = 0.0
    track_offset: float
    music_delay: float
    loop: Loop | None = None
    speech_regions: list[Region] = Field(default_factory=list)
    fade_in: float
    fade_out: float

    # Level targets (inputs to the mix).
    mix_role: Literal["underscore", "featured"] = "underscore"
    music_lu_below_dialogue: float = 12.0
    duck_db: float = -14.0
    enforce_speech_margin: bool = True
    original_gain_db: float = 0.0
    target_lufs: float = -14.0
    true_peak_dbtp: float = -1.0
    attack_ms: float = 80.0
    release_ms: float = 300.0

    # Resolved by the mix stage from the actual audio.
    dialogue_lufs: float | None = None
    music_lufs: float | None = None
    music_gain_db: float | None = None
    region_duck_db: list[float] = Field(default_factory=list)
    mix_gain_db: float | None = None
    limiter_max_reduction_db: float | None = None
    # Dialogue minus music (outside speech) after final loudness and limiting.
    final_balance_lu: float | None = None

    sfx: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Check(BaseModel):
    name: str
    passed: bool
    value: float | None = None
    target: str
    detail: str = ""


class VerifyReport(BaseModel):
    status: Literal["ok", "verify_failed", "render_failed"] = "ok"
    passed: bool
    checks: list[Check]
    skipped: list[str] = Field(default_factory=list)
    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None
    attribution: str | None = None
    licence_warnings: list[str] = Field(default_factory=list)
