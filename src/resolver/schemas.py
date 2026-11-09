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


# --- Library, brief and retrieval (M5 to M7) ---------------------------------

Licence = Literal["cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "pd", "own"]


class TrackRecord(BaseModel):
    track_id: str                 # "<adapter>:<id>"
    title: str
    artist: str
    duration: float
    path: str                     # local cached audio
    source_url: str | None = None
    licence: Licence
    licence_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    bpm: float | None = None
    vocal_prob: float | None = None
    lufs: float | None = None


class NamedReference(BaseModel):
    raw: str
    canonical: str | None = None
    kind: Literal["show", "film", "song", "game", "artist", "trend", "unknown"] = "unknown"
    available: bool = False       # always False in the prototype
    confidence: float = 0.0
    sources: list[str] = Field(default_factory=list)


class CueSpec(BaseModel):
    label: str
    cue_type: Literal["bed", "stinger", "build", "loop", "ending"] = "bed"
    mood: list[str] = Field(default_factory=list)
    energy: Literal["low", "medium", "high"] = "medium"
    tempo_bpm: tuple[int, int] | None = None
    instruments: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    vocals: Literal["none", "ok", "required"] = "none"
    mix_role: Literal["underscore", "featured"] = "underscore"
    queries: list[str] = Field(default_factory=list)


class Brief(BaseModel):
    version: str
    mode: Literal["online", "offline"]
    music_requested: Literal["explicit", "implied", "none"] = "explicit"
    named_reference: NamedReference | None = None
    primary: CueSpec
    alternatives: list[CueSpec] = Field(default_factory=list, max_length=2)
    deferred_cues: list[str] = Field(default_factory=list)
    sfx_wanted: bool = False
    ignored_instructions: list[str] = Field(default_factory=list)
    notes_for_user: list[str] = Field(default_factory=list)

    def cues(self) -> list[CueSpec]:
        return [self.primary, *self.alternatives]


class Candidate(BaseModel):
    rank: int
    track: TrackRecord
    segment_start: float
    cue_label: str
    score: float
    z: float
    reason: str
    warnings: list[str] = Field(default_factory=list)
    attribution: str | None = None
    preview_path: str | None = None


class ResolveResult(BaseModel):
    status: Literal["ok", "no_match", "needs_input", "error"]
    brief: Brief | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    searched: dict = Field(default_factory=dict)
    fallback_option: str = "licensed"
    coverage_warnings: list[str] = Field(default_factory=list)
    message: str = ""
    quoted_cost: float | None = None
