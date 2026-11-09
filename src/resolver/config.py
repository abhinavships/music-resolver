"""Tunable values. Defaults live here; `resolver.example.toml` mirrors them (a test checks).

Load a user file with `load(path)`. Keys missing from the file keep their defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LibraryConfig(_Section):
    profile: Literal["demo", "commercial"] = "demo"
    adapters: list[str] = ["local_folder"]
    local_folder: str = "library/"
    cache_dir: str = ".cache/resolver"
    jamendo_tags: list[str] = [
        "world", "indian", "comedy", "soundtrack", "action", "percussion", "funk", "orchestral",
        "electronic", "ambient",
    ]
    jamendo_limit: int = 2000
    mtg_jamendo_root: str = "data/mtg-jamendo/"


class IndexConfig(_Section):
    dir: str = "index/"
    segment_s: float = 10.0
    hop_s: float = 5.0
    max_track_s: float = 360.0
    min_segment_dbfs: float = -45.0
    coverage_threshold: float = 0.25
    coverage_instruments: list[str] = [
        "dhol", "tabla", "harmonium", "brass", "shehnai", "sitar", "taiko", "orchestral", "synth",
        "guitar", "piano",
    ]


class EmbedConfig(_Section):
    model: str = "laion/clap-htsat-unfused"
    batch_size: int = 16
    zero_shot_scale: float = 100.0


class BriefConfig(_Section):
    model: str = "claude-sonnet-5-5"
    web_search_tool: str = "web_search_20260209"
    web_search_max_uses: int = 3
    prompt_version: str = "v1"
    min_confidence_shown: float = 0.6
    server_fallback: bool = True


class RetrieveConfig(_Section):
    per_query_top_k: int = 200
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    vocal_penalty: float = 0.15
    tempo_penalty_per_10bpm: float = 0.1
    tempo_penalty_max: float = 0.3
    length_penalty: float = 0.2
    max_per_artist: int = 1
    coverage_min_segments: int = 50
    audio_query_weight: float = 3.0


class NoMatchConfig(_Section):
    z_threshold: float = 2.5
    background_samples: int = 500


class HitsConfig(_Section):
    min_score: float = 0.35
    motion_width_px: int = 160
    min_spacing_s: float = 0.5
    top_k: int = 5
    # Salience: a curve's peak counts fully only if it stands out from the curve's
    # typical level (robust z-score), so a clip with no clear event scores low.
    salience_z_low: float = 20.0
    salience_z_high: float = 60.0
    motion_floor_px: float = 0.25


class BeatsConfig(_Section):
    engine: Literal["beat_this", "librosa"] = "beat_this"
    checkpoint: str = "final0"
    beats_per_bar: int = 4
    min_confidence: float = 0.3
    onset_floor_db: float = 1.0
    onset_full_db: float = 5.0


class VadConfig(_Section):
    engine: Literal["silero", "energy"] = "silero"
    silero_threshold: float = 0.5
    silero_pad_ms: float = 100.0
    min_region_s: float = 0.2
    max_gap_s: float = 0.3


class FitConfig(_Section):
    allow_stretch: bool = False
    max_stretch: float = 0.04
    loop_crossfade_ms: float = 30.0
    short_clip_s: float = 3.0
    long_clip_s: float = 90.0
    fade_in_s: float = 0.05
    fade_out_max_s: float = 0.5


class MixConfig(_Section):
    target_lufs: float = -14.0
    true_peak_dbtp: float = -1.0
    underscore_lu_below_dialogue: float = 12.0
    featured_lu_below_dialogue: float = 4.0
    featured_max_speech_fraction: float = 0.25
    no_speech_reference_lufs: float = -23.0
    duck_db: float = -14.0
    min_speech_margin_db: float = 10.0
    speech_margin_safety_db: float = 1.0
    attack_ms: float = 80.0
    release_ms: float = 300.0
    limiter_lookahead_ms: float = 5.0
    balance_warn_lu: float = 2.0
    music_in_source_threshold: float = 0.30
    original_gain_db_when_music: float = -6.0


class SfxConfig(_Section):
    enabled: bool = False
    gain_db: float = -6.0
    folder: str = "sfx/"
    max_effects: int = 2
    min_hit_score: float = 0.2


class GenerationConfig(_Section):
    provider: str = "none"


class SessionConfig(_Section):
    preview_height: int = 360
    different_neighbours: int = 20


class VerifyConfig(_Section):
    max_hit_offset_s: float = 0.08
    max_av_duration_diff_s: float = 0.1
    lufs_tolerance_lu: float = 1.0
    music_present_dbfs: float = -60.0


class Config(_Section):
    library: LibraryConfig = LibraryConfig()
    index: IndexConfig = IndexConfig()
    embed: EmbedConfig = EmbedConfig()
    brief: BriefConfig = BriefConfig()
    retrieve: RetrieveConfig = RetrieveConfig()
    no_match: NoMatchConfig = NoMatchConfig()
    hits: HitsConfig = HitsConfig()
    beats: BeatsConfig = BeatsConfig()
    vad: VadConfig = VadConfig()
    fit: FitConfig = FitConfig()
    mix: MixConfig = MixConfig()
    sfx: SfxConfig = SfxConfig()
    generation: GenerationConfig = GenerationConfig()
    session: SessionConfig = SessionConfig()
    verify: VerifyConfig = VerifyConfig()


DEFAULT_FILE = "resolver.toml"


def load(path: str | Path | None = None) -> Config:
    """Read `path`, or `./resolver.toml` if it exists, else return the defaults.

    Unknown keys raise, so a typo in the file is caught instead of ignored.
    """
    if path is None:
        candidate = Path(DEFAULT_FILE)
        if not candidate.exists():
            return Config()
        path = candidate
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return Config.model_validate(data)
