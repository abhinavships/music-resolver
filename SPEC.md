# Build spec: music-resolver

Turn a music reference in an editing prompt into three licensed, previewable tracks, then fit the chosen one to the clip and prove the result with measured checks.

The prototype answers one question: **does searching a decomposed brief beat searching the raw prompt?** Everything else exists to make that test fair and the demo usable.

Written 2026-11-08. Milestones M1 to M3 were built when this was written; M4 to M9 were built on 2026-11-09 (see section 1 and README "Known limits").

---

## 0. Rules for whoever builds this

1. Finish milestones in order. Each one ends with `pytest` green and its acceptance tests added.
2. Never download, stream or place copyrighted audio the user named. Reference names are used only for text lookup.
3. Filter by licence **before** ranking. A track with an unknown licence never reaches a user.
4. Every stage reads and writes JSON artifacts (`schemas.py`), so a stage can be rerun or tested alone.
5. Every degraded path (no key, no network, no model) still produces a result and says what was skipped.
6. No silent magic numbers: tunable values live in `resolver.toml` (see `resolver.example.toml`), with the defaults below.
7. Synthetic tests stay; real-media tests skip cleanly when fixtures or models are missing (`pytest.mark.skipif`).
8. Keep the user's video local. Only prompt text (and later, optionally, a transcript) goes to the LLM.

---

## 1. Current state (M1 to M9)

| Module | Does | Known gaps |
| --- | --- | --- |
| `config.py` | Defaults for every tunable value; loads `resolver.toml`, rejects unknown keys | — |
| `fallback.py` | Option order, skip, no-match, charge approval | No "music slot" export on the last option (L9, M7) |
| `clip.py` | Hits from audio onsets plus motion, timed by frame timestamps (C5) and stream offsets (C6), scaled by salience (C7) | Salience thresholds calibrated on synthetic media |
| `vad.py` | Silero VAD with the energy-band fallback | — |
| `beats.py` | Beat This! beats and downbeats, librosa fallback (D4), pulse confidence (T5) | librosa bar lines are estimated from accents |
| `fit.py` | Downbeat (or beat) aligned to the main hit (F1); no-hit (C7), short-clip (C9) and long-clip (C10) paths | Ending on a downbeat (F5) not built |
| `segment.py` | Loops whole bars with an equal-power crossfade, snapped to the fitted beat period (T3) | Other segments of the same track (M7) |
| `mixing.py` | Loudness-normalised music (T1), role levels (F2, F3), attack/release ducking with a per-region margin (F4), -14 LUFS / -1 dBTP mix (F6) | Source music detection (C3) |
| `render.py` | Mix on the clip timeline, mux with the original video stream | — |
| `verify.py` | A/V end times, music present, clipping, LUFS, true peak, speech masking, hit offset (same engine as the plan) | Measured on pre-encode stems |
| `library/` | Licence parsing and profile filter (L1 to L6); `local_folder`, `jamendo_api`, `mtg_jamendo` adapters | Jamendo and MTG adapters tested on recorded data only |
| `embed.py`, `ingest.py` | CLAP segment embeddings (default `laion/clap-htsat-unfused`), tag-only fallback (D3), vocal_prob, bpm, LUFS, coverage, idempotent | `larger_clap_music` is degenerate as published |
| `brief.py` | Claude + web search brief, rights guard, offline keyword map, retry (D5), cache (D8), refine | Live call untested here (no key) |
| `retrieve.py` | Multi-query RRF, penalties (T2, T6, length), z no-match, slots (R3), one per artist (B5), MMR, reasons, coverage warnings (B4) | z threshold untuned |
| `session.py`, `sfx.py`, `generation.py` | resolve / refine / pick / skip / own / generate, exclusions (R11 to R13), music-slot export (L9), effects layer (F7), generation interface | No generation provider |
| `review.py`, `tool.py`, `mcp_server.py` | Review page with ratings export, blind eval pages, JSON tool contract, MCP wrapper | — |
| `evaluate.py` | Conditions A/B/C, ratings, Krippendorff's alpha, threshold split, report | Needs labels, hand briefs, raters |
| `cli.py` | `analyse`, `fit`, `index`, `brief`, `resolve`, `refine`, `pick`, `skip`, `own`, `eval`, `tool` | — |

135 tests pass and 3 skip (the M4 real-clip tests, until `fixtures/real/labels.json` exists) with the `ml`, `api`, `ui` and `mcp` extras installed. Retrieval tests use a synthetic toy library and a tone-based stand-in embedder; `pytest -m live` runs real CLAP and, with a key, a live brief.

---

## 2. Scenarios and required behaviour

Each row is a requirement. The ID is referenced by tests (`test_<id>_...`).

### 2.1 Requests

| ID | Scenario | Behaviour |
| --- | --- | --- |
| R1 | Named show, film, song or game ("Tarak Mehta fight music") | Brief stage looks the name up with web search, writes a sound description (instruments, tempo, energy, cue type) with sources and a confidence, and sets `named_reference.available = false`. The UI says once that the original is unavailable, then continues. |
| R2 | Misspelt or transliterated name ("tarak mehta ka ulta chasma") | Lookup normalises to the canonical name. Confidence < 0.6 is shown in the brief header. |
| R3 | Two readings in one ask ("famous fighting or long sequence music") | `brief.cues` holds one primary and up to two `alternatives`. Top three previews cover the alternatives (at least one slot each when a match exists), labelled. |
| R4 | Mood or genre only | No web lookup. Brief built from the words. |
| R5 | Artist style ("Hans Zimmer style") | Brief describes genre, instruments, energy. Rights guard removes artist and title tokens from search queries. |
| R6 | Reference audio file supplied as a reference | Extract tempo and a CLAP audio embedding; search audio-to-audio and fuse with text results. Never placed in output unless the user later chooses it as own audio (R12). |
| R7 | Link (YouTube, Instagram) | Treated as R1 using the link text and any title found by search. No download in the prototype. |
| R8 | Prompt mixes music with other edits ("add caption ... and music") | Brief extracts only the music request; `brief.ignored_instructions` lists the rest. |
| R9 | Sound effects wanted, or implied by a fight/comedy cue | Optional effects layer: CC0 effects placed on secondary hits (M7, behind `sfx.enabled`). |
| R10 | Music implied, not stated ("make it funny") | Brief marks `music_requested = "implied"`; resolver proposes music but does not auto-apply it. |
| R11 | Follow-up: "more comic", "no drums", "faster" | `refine` edits the brief (delta, not rewrite), re-searches, excludes tracks already shown unless asked. |
| R12 | Follow-up: "like #2" | Audio-to-audio search from candidate 2's segment embedding. |
| R13 | Follow-up: "something completely different" | Exclude shown tracks and their nearest 20 neighbours. |
| R14 | Non-English prompt | Brief stage handles it; search queries are always English. |
| R15 | Request for multiple cues in one clip ("dramatic then comic") | v1: one cue; brief records the second as `deferred_cues` and the report says so. |

### 2.2 Rights and sourcing

| ID | Scenario | Behaviour |
| --- | --- | --- |
| L1 | CC0 / public domain | Allowed in every profile. |
| L2 | CC BY | Allowed. Report includes a ready-to-paste attribution line (title, artist, licence URL, source URL). |
| L3 | CC BY-SA | `demo` profile only, warning shown: the edit may have to carry the same licence. |
| L4 | Non-commercial (NC) | `demo` profile only, warning shown. MTG-Jamendo as a whole is research and non-commercial only, so the demo profile is the default for the prototype. |
| L5 | No-Derivatives (ND) | Always excluded. CC 4.0 states adapted material is always produced when a music track is synched in timed relation with a moving image. |
| L6 | Licence missing, unknown or unparseable | Excluded at ingest, counted in the coverage report. |
| L7 | User's own file | Allowed after the user confirms rights (CLI flag `--i-have-rights` or UI checkbox). Recorded in the report. |
| L8 | "Make it sound exactly like X" / melody description | Brief never contains melody, hook, or lyric descriptions. The UI offers a genre-level match instead. |
| L9 | Named track unavailable but may exist in the posting app | Last fallback option exports the edit without music plus `hit_marker.json` and the hit time in the report, so the user can add the in-app sound and line it up. |

### 2.3 Library outcomes

| ID | Scenario | Behaviour |
| --- | --- | --- |
| B1 | Strong matches | Top three by fused score, made diverse with MMR (λ = 0.7), each with a one-line reason built from matched brief attributes, plus licence. |
| B2 | Weak matches | Candidates below `no_match.threshold` are dropped; if fewer than one remains, status `no_match`. |
| B3 | No match | Status `no_match`, `searched` lists the queries and library size, fallback advances to own file. |
| B4 | Regional coverage gap (few dhol/tabla tracks) | Coverage report at index time counts segments per target tag; resolver warns when the brief's key instruments have fewer than 50 segments. |
| B5 | Same artist dominating results | Max one track per artist in the top three. |

### 2.4 Source clip

| ID | Scenario | Behaviour |
| --- | --- | --- |
| C1 | Dialogue | Silero VAD regions; ducking under them. |
| C2 | Shouting, laughter, crowd | Counted as dialogue for ducking (VAD output is accepted as is). |
| C3 | Clip already contains music | CLAP zero-shot on 2 s windows ("music", "speech", "crowd noise", "silence"). If music > 30 % of the clip: warn, default `original_gain_db = -6`, options keep / lower / proceed. |
| C4 | No audio track | Skip VAD and audio onsets; hits from motion only; render adds an audio stream. (Already handled; keep the test.) |
| C5 | Variable frame rate (WhatsApp, phone) | Motion timestamps from `CAP_PROP_POS_MSEC`. Test with a VFR fixture; hit error < 0.1 s. |
| C6 | Non-zero stream start time | Read `start_time` from ffprobe and offset audio and video analysis. |
| C7 | No clear hit (top hit score < `hits.min_score`, default 0.35) | No hit sync: start the music on a downbeat at 0 s; report `hit_sync = "none"`. |
| C8 | Several strong hits | Align the main hit; report offsets of the next two; SFX layer (R9) may cover them. |
| C9 | Clip shorter than 3 s | Prefer candidates whose segment has an onset peak in the first 0.5 s (stinger behaviour); no fade-in. |
| C10 | Clip longer than 90 s | One cue, warning in the report. Hit sync still applies to the main hit. |
| C11 | User overrides hit or speech regions | CLI flags already exist; keep them and expose them in the agent contract. |

### 2.5 Track

| ID | Scenario | Behaviour |
| --- | --- | --- |
| T1 | Loudness varies by master | Normalise the placed segment to `mix.music_lufs` before applying role gains (pyloudnorm). |
| T2 | Vocals | Ingest stores `vocal_prob` (CLAP zero-shot "singing voice" vs "instrumental"). Ranking penalty `0.15 × vocal_prob` when the clip has dialogue. |
| T3 | Track shorter than needed from the chosen offset | First try other segments of the same track; otherwise loop whole bars (from downbeats) with a 30 ms equal-power crossfade. |
| T4 | Long track with an intro | Search is over 10 s segments (hop 5 s), so the matching part is used, not the start of the file. |
| T5 | No reliable beats (ambient) | Beat confidence below threshold: no hit sync, start at the best segment, report `hit_sync = "none"`. |
| T6 | Tempo far from the brief's range | Soft penalty; never time-stretch more than ±4 %, and only when `fit.allow_stretch` is on (off by default). |
| T7 | Mono, odd sample rate, corrupt file | ffmpeg decode normalises format; corrupt files are skipped at ingest and counted. |

### 2.6 Fit and mix

| ID | Scenario | Behaviour |
| --- | --- | --- |
| F1 | Normal | Align a **downbeat** (Beat This!) to the main hit; fall back to the strongest librosa beat. Choose the segment start so the music covers the clip. |
| F2 | Mix role: underscore (talky clip) | Music sits `mix.underscore_lu_below_dialogue` (default 12 LU) below dialogue loudness outside speech, and is ducked further inside speech. |
| F3 | Mix role: featured (montage, fight, little speech) | Music `mix.featured_lu_below_dialogue` (default 4 LU) below dialogue; ducking still keeps the 10 dB speech margin. |
| F4 | Duck timing | Attack 80 ms (music already down when speech starts), release 300 ms. |
| F5 | Ending | Fade out over `min(0.5, duration / 10)` s; if a downbeat falls in the last 0.4 s, end there with a 60 ms fade instead. |
| F6 | Final level | Mix normalised to `mix.target_lufs` (default −14 LUFS integrated) with true peak ≤ −1 dBTP. These are common social-video targets, kept configurable. |
| F7 | Effects layer | When enabled: one effect per secondary hit, peak-aligned to the hit, level `sfx.gain_db`. |

### 2.7 Failures and degraded modes

| ID | Scenario | Behaviour |
| --- | --- | --- |
| D1 | No `ANTHROPIC_API_KEY` | Offline brief: keyword and tag map plus the raw prompt as one query; `brief.mode = "offline"`; shown in output. |
| D2 | No network | Local library only; no web lookup; same as D1 for the brief. |
| D3 | CLAP model unavailable | Tag search over metadata (BM25 on tags and titles); status notes it. |
| D4 | Beat This! unavailable | librosa beats, no downbeats; report notes it. |
| D5 | LLM returns invalid JSON | One retry with the validation error appended; then offline brief. |
| D6 | Render fails | Keep artifacts, write `report.json` with `status = "render_failed"` and the ffmpeg error tail. |
| D7 | Verification fails | Exit code 1, report lists failing checks; the session still offers the next candidate. |
| D8 | Same prompt again | Brief cached by hash of (normalised prompt, model, brief prompt version). |

---

## 3. Decisions

| Fork | Choice | Kept behind config |
| --- | --- | --- |
| Brief model | Claude API, `claude-sonnet-5-5`, server web search tool `web_search_20250305` with `max_uses = 3` | `brief.model`, `brief.web_search_tool`. Check the current tool versions on the [web search tool page](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) before building; newer versions exist. Searches cost $10 per 1,000 on top of tokens. |
| Brief output | JSON validated by pydantic `Brief`; on failure one retry | — |
| Embedding | `laion/larger_clap_music` via `transformers` `ClapModel` / `ClapProcessor` (Apache-2.0). Read the sample rate from `processor.feature_extractor.sampling_rate`; do not hardcode it. | `embed.model` (e.g. `laion/clap-htsat-unfused` for comparison) |
| Index granularity | 10 s segments, 5 s hop, max 6 min per track | `index.segment_s`, `index.hop_s` |
| Vector store | FAISS `IndexFlatIP` on L2-normalised vectors (small library, exact search) | — |
| Query fusion | Reciprocal rank fusion (k = 60) across brief queries, and with audio-to-audio results when present | `retrieve.rrf_k` |
| Re-rank | Fused score − vocal penalty − tempo penalty; one track per artist; MMR for diversity | weights in config |
| No-match | Per-query z-score of the best segment against 500 random library segments; threshold tuned on the eval set's matchable vs unmatchable labels (initial 2.5) | `no_match.z_threshold` |
| Beats | Beat This! `File2Beats(checkpoint_path="final0", dbn=False)` (MIT), device order cuda → mps → cpu; librosa fallback | `beats.engine` |
| VAD | Silero VAD; the energy-band detector remains as fallback | `vad.engine` |
| Loudness | pyloudnorm | — |
| Library sources | Adapters: `local_folder` (audio + `licences.csv`), `jamendo_api` (needs a free `client_id`; respects `audiodownload_allowed`), `mtg_jamendo` subset | `library.adapters`, `library.profile` (`demo` or `commercial`) |
| Effects source | Freesound API, CC0 only, cached locally | `sfx.enabled` (off by default) |
| Generation | Interface only (`GenerationProvider.quote()`, `.generate()`), no provider in v1 | `generation.provider = "none"` |
| UI | Static review page generated per session; no server needed | — |
| Agent integration | JSON-in, JSON-out CLI (`resolver tool ...`); optional MCP server in M8 | — |

---

## 4. Architecture

```
prompt ─▶ brief.py ─▶ brief.json ──────────────┐
                                               ▼
library adapters ─▶ ingest.py ─▶ index/ ─▶ retrieve.py ─▶ candidates.json
                                               ▲               │
clip.mp4 ─▶ clip.py ─▶ clip.json ──────────────┘               ▼
                                     session.py (state, refine, fallback)
                                                               │ pick
                                                               ▼
                                 segment.py ─▶ fit.py ─▶ mix.json ─▶ render.py ─▶ output.mp4
                                                                          │
                                                                          ▼
                                                                 verify.py ─▶ report.json
```

Clip analysis feeds retrieval because dialogue presence changes the vocal penalty and mix role, and clip length decides segment fit.

### New and changed modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Load `resolver.toml`, defaults, profile |
| `brief.py` | Prompt to `Brief`; online (Claude + web search) and offline modes; rights guard; cache |
| `library/base.py` | `TrackRecord`, adapter protocol, licence parsing and filtering |
| `library/local_folder.py`, `library/jamendo_api.py`, `library/mtg_jamendo.py` | Adapters |
| `ingest.py` | Decode, segment, embed, tag (vocal_prob, bpm, loudness), write index and coverage report |
| `embed.py` | CLAP text and audio embeddings, batched, cached on disk |
| `retrieve.py` | Multi-query search, fusion, re-rank, MMR, no-match decision, reasons |
| `segment.py` | Choose the segment start and handle looping for a chosen track |
| `beats.py` | Beat This! wrapper with librosa fallback; returns beats, downbeats, confidence |
| `vad.py` | Silero VAD wrapper with the energy-band fallback |
| `mixing.py` | Loudness normalisation, role levels, ducking envelope with separate attack and release, final loudness |
| `session.py` | Session state on disk, refine, exclusions, fallback integration |
| `sfx.py` | Effects layer (M7, optional) |
| `review.py` | Static review page with previews and rating capture |
| `tool.py` | Agent tool contract (section 6.3) |
| `eval/run.py` | Conditions A, B, C; metrics; report |

`fit.py`, `clip.py`, `render.py` and `verify.py` are extended, not replaced.

---

## 5. Schemas (add to `schemas.py`)

```python
class NamedReference(BaseModel):
    raw: str                      # as the user wrote it
    canonical: str | None         # after lookup
    kind: Literal["show", "film", "song", "game", "artist", "trend", "unknown"]
    available: bool = False       # always False in the prototype
    confidence: float             # 0..1
    sources: list[str] = []       # URLs the lookup used

class CueSpec(BaseModel):
    label: str                    # "comic fight underscore"
    cue_type: Literal["bed", "stinger", "build", "loop", "ending"]
    mood: list[str]
    energy: Literal["low", "medium", "high"]
    tempo_bpm: tuple[int, int] | None
    instruments: list[str]
    avoid: list[str] = []
    vocals: Literal["none", "ok", "required"] = "none"
    mix_role: Literal["underscore", "featured"]
    queries: list[str]            # 3-5 CLAP-friendly English phrasings, rights-guarded

class Brief(BaseModel):
    version: str                  # brief prompt version, part of the cache key
    mode: Literal["online", "offline"]
    music_requested: Literal["explicit", "implied", "none"]
    named_reference: NamedReference | None
    primary: CueSpec
    alternatives: list[CueSpec] = []   # max 2 (R3)
    deferred_cues: list[str] = []      # R15
    sfx_wanted: bool = False
    ignored_instructions: list[str] = []
    notes_for_user: list[str] = []     # e.g. "The original theme is unavailable; matching its sound."

class TrackRecord(BaseModel):
    track_id: str                 # "<adapter>:<id>"
    title: str
    artist: str
    duration: float
    path: str                     # local cached audio
    source_url: str | None
    licence: Literal["cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "pd", "own"]
    licence_url: str | None
    tags: list[str] = []
    bpm: float | None
    vocal_prob: float | None
    lufs: float | None

class Candidate(BaseModel):
    rank: int
    track: TrackRecord
    segment_start: float
    cue_label: str                # which CueSpec it answers (R3)
    score: float
    z: float
    reason: str                   # "matches: dhol, tabla, comic, fast"
    warnings: list[str] = []      # licence warnings (L3, L4), vocals, tempo
    attribution: str | None       # L2
    preview_path: str | None

class ResolveResult(BaseModel):
    status: Literal["ok", "no_match", "needs_input", "error"]
    brief: Brief
    candidates: list[Candidate]
    searched: dict                # queries, library size, filters applied
    fallback_option: str          # current Fallback option
    coverage_warnings: list[str] = []
    message: str                  # one plain sentence for the user
```

Extend `MixPlan` with `segment_start`, `loop_points`, `mix_role`, `original_gain_db`, `music_lufs`, `downbeat_used: bool`, `hit_sync: Literal["downbeat", "beat", "none"]`, `sfx: list[dict]`.

Extend `VerifyReport` with `status`, `integrated_lufs`, `true_peak_dbtp`, `attribution`, `licence_warnings`.

---

## 6. Interfaces

### 6.1 CLI

```
resolver index  --config resolver.toml            # ingest adapters, build index, write coverage.json
resolver brief  "prompt"                          # print brief.json
resolver resolve clip.mp4 "prompt" [--reference ref.wav] [--out out/]   # brief + analysis + top 3 + previews + review.html
resolver refine  out/ "more comic"                # R11-R13
resolver pick    out/ 2 [--hit 7.2] [--speech ...] # fit, render, verify the chosen candidate
resolver skip    out/ [--reason "..."]            # advance the fallback
resolver own     out/ track.mp3 --i-have-rights   # L7
resolver analyse clip.mp4                         # existing
resolver fit clip.mp4 track.mp3                   # existing, kept for direct use
resolver eval   --prompts eval/prompts.jsonl --conditions A,B,C
resolver tool                                     # JSON on stdin, JSON on stdout (6.3)
```

All commands accept `--config`. Exit codes: 0 ok, 1 verification failed, 2 no match, 3 bad input, 4 dependency missing (with the degraded mode still applied where possible).

### 6.2 Session folder

```
out/
  session.json        # brief, shown track ids, fallback state, history of refinements
  clip.json  brief.json  candidates.json
  previews/1.mp4 2.mp4 3.mp4      # 360p, fast encode, candidate mixed under the clip
  review.html
  mix.json  report.json  output.mp4  stems/
  hit_marker.json     # L9 only
```

### 6.3 Agent tool contract

One JSON request per call, so Cardboard's Director, an MCP client or a test can drive it.

```json
{"action": "resolve", "clip": "clip.mp4", "prompt": "tarak mehta fight music", "reference": null, "session": "out/"}
{"action": "refine",  "session": "out/", "instruction": "more comic, less brass"}
{"action": "pick",    "session": "out/", "rank": 2, "overrides": {"hit": 7.2}}
{"action": "skip",    "session": "out/", "reason": "none fit"}
{"action": "own",     "session": "out/", "track": "song.mp3", "rights_confirmed": true}
```

Every response is a `ResolveResult` (resolve, refine, skip) or `VerifyReport` plus output paths (pick, own). `generate` responds with `status = "needs_input"` and the quoted cost; it never charges inside a call that did not carry `approved_cost` equal to the quote.

M8 may wrap the same handler in an MCP server with tools `resolve_music`, `refine_music`, `pick_music`, `skip_music`. Do this only after the CLI contract tests pass.

---

## 7. Stage details

### 7.1 Brief (`brief.py`)

System prompt rules (store as `brief_prompt_v1.txt`, version in the cache key):

- Extract only the music request; list other instructions in `ignored_instructions`.
- If a named reference exists, search the web for how its music sounds (instruments, tempo feel, era, typical use) and cite sources. Do not describe melody, hook or lyrics.
- Disjunctions become `alternatives` (max 2).
- `queries`: 3 to 5 short English phrasings a music-text model understands ("fast comic Indian percussion with dhol and tabla, slapstick fight"), no proper nouns.
- Output only JSON matching the schema.

Rights guard, applied after the model and in tests:

- Remove any token from `named_reference.raw`, `canonical` or detected artist names from every query.
- Reject queries containing "melody", "hook", "lyrics", "same tune", "exactly like"; regenerate once, else drop the query.
- At least two queries must survive, else fall back to offline queries.

Offline brief (D1, D2): lowercase the prompt, map keywords with `brief_keywords.toml` (e.g. `fight → action, percussion, high energy`; `comic → playful, quirky`; `tarak mehta → indian tv comedy, dhol, tabla, harmonium`), and use the raw prompt as one extra query.

### 7.2 Ingest (`ingest.py`)

1. Each adapter yields `TrackRecord` stubs. Drop L5 and L6 immediately; drop L3 and L4 unless profile is `demo`.
2. Decode with ffmpeg to mono at the CLAP sample rate; skip corrupt files (T7).
3. Segments: 10 s windows, 5 s hop, up to 6 min; drop segments with RMS below −45 dBFS.
4. Embed segments in batches; store `float16` vectors plus `(track_id, start)` in `index/segments.npy` and `index/meta.parquet` (or JSONL if pyarrow is unavailable).
5. Per track: `bpm` (librosa), `lufs` (pyloudnorm), `vocal_prob` (mean CLAP zero-shot over segments).
6. Coverage report `index/coverage.json`: tracks and segments per licence, per tag, and per target instrument list (`dhol`, `tabla`, `harmonium`, `brass`, `shehnai`, `sitar`, `taiko`, `orchestral`, `synth`, `guitar`, `piano`), measured by zero-shot similarity above a set threshold.
7. Idempotent: re-running skips tracks already embedded with the same model id.

Start small: 1,000 to 3,000 tracks chosen by tags relevant to the prompt set (world, indian, comedy, soundtrack, action, percussion, funk, orchestral, electronic, ambient). Measure coverage before scaling.

### 7.3 Retrieve (`retrieve.py`)

1. Embed each query of each CueSpec (and the reference audio if given).
2. Top 200 segments per query; RRF across queries within a cue.
3. Collapse to tracks, keeping each track's best segment.
4. Penalties: vocals (T2), tempo outside range (0.1 per 10 BPM outside, max 0.3), clip-length fit (segment start + clip length beyond track end without a loop, 0.2).
5. No-match: z-score of the best raw similarity for each query versus 500 random segments; a cue is matched if the median z across its queries ≥ `no_match.z_threshold`.
6. Select three: allocate slots across primary and alternatives (R3), one per artist (B5), MMR within each cue.
7. Reason: the brief attributes whose zero-shot similarity on the chosen segment is in the top quartile for that attribute ("dhol, fast, playful").
8. Exclusions from the session (R11 to R13) apply before step 6.

### 7.4 Segment and fit (`segment.py`, `fit.py`)

- Beats and downbeats on the candidate's best segment plus 20 s either side.
- Choose the downbeat nearest the segment's best part that satisfies: downbeat − hit ≥ 0 and the track covers the clip from the resulting offset. Otherwise try other segments of the track with similarity within 0.05 of the best. Otherwise loop (T3).
- No hit (C7) or no reliable beats (T5): start at a downbeat (or the segment start) at 0 s.
- Short clip (C9): prefer a segment start on its strongest onset.

### 7.5 Mix (`mixing.py`)

1. Dialogue loudness: integrated LUFS of the clip audio within VAD regions (if no speech, use −23 LUFS as the reference).
2. Music target = dialogue LUFS − role offset (F2, F3); normalise the placed music to it.
3. Ducking envelope with attack 80 ms, release 300 ms; the duck depth is the larger of `mix.duck_db` and what is needed to keep a 10 dB speech margin, measured per region.
4. Original clip gain (C3).
5. Sum, then normalise the mix to −14 LUFS and limit true peak to −1 dBTP (simple look-ahead limiter or gain only if no peaks exceed).

### 7.6 Verify (`verify.py`)

Existing checks plus:

| Check | Target |
| --- | --- |
| `integrated_lufs` | Within ±1 LU of `mix.target_lufs` |
| `true_peak` | ≤ −1 dBTP |
| `speech_masking_encoded` | Re-measure masking on the encoded output against the stems (correlation-based estimate is enough), within 2 dB of the pre-encode value |
| `hit_offset` | Downbeat used: ≤ 80 ms; beat: ≤ 80 ms; none: skipped with reason |
| `licence_allowed` | Licence permitted by the profile at render time |
| `attribution_present` | Required for CC BY and BY-SA |

### 7.7 Review page (`review.py`)

One static HTML file per session: the brief (editable fields shown read-only, with the command to refine), three `<video>` previews, reasons, licences and warnings, and a 1 to 5 rating per preview saved by a "Download ratings" button as JSON (no server). The eval mode generates the same page with condition labels hidden and order shuffled per rater.

---

## 8. Milestones and acceptance tests

Each milestone lists the tests that must exist and pass.

### M3 Hardening (no network needed)

- VFR timing (C5): fixture made with ffmpeg `setpts` producing variable frame intervals; `test_c5_vfr_hit_within_100ms`.
- Stream start offset (C6): fixture with `-itsoffset`; `test_c6_start_time_offset`.
- Loudness normalisation (T1, F2, F3, F6): the same track at −30 and −8 LUFS master gives music levels within 1 LU; `test_t1_level_independent_of_master`, `test_f6_final_lufs_and_peak`.
- Attack and release (F4): `test_f4_duck_attack_release`.
- Looping (T3): 6 s track on a 12 s clip loops whole bars with no click (max sample jump at the joint < 0.05); `test_t3_loop_whole_bars`.
- Beats engine wrapper with fallback (D4): `test_d4_librosa_fallback_when_beat_this_missing` (monkeypatch import).
- Beat This! downbeat alignment when installed: `test_f1_downbeat_alignment` (skip if missing).
- Silero VAD wrapper with fallback: synthetic speech test still passes with either engine.
- No-hit path (C7) and short clip (C9).
- `resolver.example.toml` loaded by `config.py`; every number in this spec's defaults lives there.

### M4 Real clip

- `fixtures/real/README.md` describes the expected files: Abhinav's 12.9 s reel and one licensed track, with hand labels in `labels.json` (main hit time, speech regions).
- `test_real_clip_labels` (skipped without fixtures): hit within 0.15 s of the label, speech recall ≥ 0.8.
- Manual listening note added to `fixtures/real/NOTES.md`.

### M5 Library and index

- Adapters with licence parsing; `test_l5_nd_always_excluded`, `test_l6_unknown_licence_excluded`, `test_l3_l4_demo_only`.
- `local_folder` adapter tested with generated tone files and a `licences.csv`.
- `jamendo_api` adapter tested with recorded JSON responses (no network in tests); live run documented in README.
- Ingest idempotency test; coverage report written.

### M6 Brief

- Pydantic schema, prompt file, rights guard, cache.
- `test_l8_rights_guard_strips_names_and_melody_words`.
- `test_r3_disjunction_gives_alternatives` and `test_r8_ignores_other_instructions` using recorded model outputs (fixtures), plus one live test marked `@pytest.mark.live`.
- `test_d1_offline_brief_without_key`, `test_d5_retry_then_offline_on_bad_json`.

### M7 Retrieval, session, effects

- Retrieval on a toy index built from synthetic tracks with known tags (embedding stub in tests; real CLAP marked `live`).
- `test_b2_b3_no_match_status`, `test_b5_one_track_per_artist`, `test_r3_slots_cover_alternatives`, `test_r11_r13_exclusions`, `test_r12_more_like_this`.
- Session flow: resolve → skip → own → pick; fallback summary in the report; `test_l9_music_slot_export`.
- Effects layer behind a flag; `test_f7_sfx_on_secondary_hits` with local CC0 fixtures.

### M8 Review page and agent tool

- `resolver tool` JSON contract tests for every action, including `generate` refusing without `approved_cost`.
- Review page renders three previews and exports ratings JSON (Playwright test, skipped if Chromium is missing).
- Optional MCP server after the contract tests pass.

### M9 Evaluation

- `eval/prompts.jsonl` (30 prompts, provided). Before scoring, an annotator searches the built library by tags for each prompt and fills `matchable: true/false` in `eval/labels.jsonl`.
- Conditions: A raw prompt as the only query; B generated brief; C hand-written brief (`eval/hand_briefs.jsonl`, to be written by Abhinav).
- Metrics and pass bars as in the PRD: top-3 usable rate (B ≥ A + 20 points), no-match accuracy ≥ 80 %, false no-match ≤ 15 %, hit offset ≤ 80 ms on 90 % of clips, speech margin ≥ 10 dB, zero dead ends.
- Report: per-condition table, rater agreement (Krippendorff's alpha), sample size, raw ratings, and a list of failures with their brief and top candidates.
- The no-match threshold is tuned on half the prompts and reported on the other half.

---

## 9. Configuration

See `resolver.example.toml`. Copy it to `resolver.toml`; keys missing from the user's file take the defaults.

## 10. Environment

- Python 3.11 to 3.13; ffmpeg on PATH.
- Optional extras in `pyproject.toml`: `ml` (torch, transformers, faiss-cpu, silero-vad, pyloudnorm, beat-this), `api` (anthropic), `ui` (playwright).
- `ANTHROPIC_API_KEY` for online briefs; `JAMENDO_CLIENT_ID` for the Jamendo adapter; `FREESOUND_API_KEY` for effects.
- First run downloads CLAP and Beat This! checkpoints; document sizes once measured.

## 11. Non-goals

- Downloading, cloning or imitating a named copyrighted track or its melody.
- Integration with Cardboard's codebase (not available); the tool contract is the integration surface.
- Multiple cues per clip (recorded as `deferred_cues`).
- Generation providers (interface only).
- Fonts, colour grades and transitions from style references.

## 12. Open decisions for Abhinav (defaults are used until changed)

| Decision | Default |
| --- | --- |
| Library profile for the public demo | `demo` (allows NC and SA, with warnings) |
| Library source for the first index | Jamendo API with a free client id, about 2,000 tracks |
| Raters for the evaluation | Three people who did not build it |
| Hand-written briefs for condition C | Written by Abhinav before seeing results |
| Effects layer in the demo | Off, turned on only if time allows |

## Sources

- [MTG-Jamendo dataset](https://mtg.github.io/mtg-jamendo-dataset): audio under per-track CC licences; dataset for non-commercial research use only.
- [laion/larger_clap_music](https://huggingface.co/laion/larger_clap_music): Apache-2.0, `ClapModel` / `ClapProcessor`.
- [beat-this on PyPI](https://pypi.org/project/beat-this/): MIT, beats and downbeats, `File2Beats`, CPU fallback.
- [Jamendo API v3](https://developer.jamendo.com/v3.0/artists/tracks): `client_id`, `license_ccurl`, `audiodownload_allowed`.
- [CC BY 4.0 legal code (copy)](https://physionet.org/about/licenses/creative-commons-attribution-40-international-public-license/): music synched to moving image always produces adapted material.
- [Claude web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool): tool versions and pricing.
- [Cardboard hard problems](https://www.cardboard.ai/hard-problems), problem 06: style references.
