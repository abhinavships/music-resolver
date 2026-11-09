# music-resolver

Turns a music reference in an editing prompt into a licensed track fitted to the clip.

**Status:** milestones M1 to M9 are built and tested offline. Every step runs without a key or network, and says what it skipped. Three things wait on you:

| Needs | For | Where |
| --- | --- | --- |
| Hand labels for the reel (`labels.json`) and a licensed track | M4 real-clip test and listening note | `fixtures/real/` |
| A music library: a folder with `licences.csv`, or a free `JAMENDO_CLIENT_ID` | Real retrieval (M5 to M7 run on a synthetic toy library in tests) | `resolver.toml` `[library]` |
| `ANTHROPIC_API_KEY` | Online briefs with web search (offline keyword briefs work without it) | environment |
| `matchable` labels, hand briefs, three raters | The M9 result | `eval/` |

## Install

```bash
pip install -e ".[dev]"              # core: librosa beats, energy-band speech, tag search
pip install -e ".[dev,ml]"           # Beat This!, Silero VAD, CLAP embeddings (torch, ~600 MB model on first use)
pip install -e ".[dev,ml,api,ui]"    # plus Claude briefs and the Playwright review-page test
playwright install chromium          # only for that test
pytest                               # offline suite; `pytest -m live` needs keys, models, network
```

## Run

```bash
cp resolver.example.toml resolver.toml          # point [library] at your music
resolver index                                  # licence filter, segment, embed, coverage.json

resolver resolve clip.mp4 "tarak mehta fight music" --out out    # brief, top 3, previews, review.html
resolver refine out "more comic, no brass"     # also "like #2", "something completely different"
resolver pick out 2 [--hit 7.2]                 # fit, render, verify -> out/output.mp4, report.json
resolver skip out --reason "none fit"           # next fallback: own file, generate, then no music + hit_marker.json
resolver own out song.mp3 --i-have-rights       # your own track (L7)

resolver brief "make it funny" --offline        # just the brief
echo '{"action":"resolve","clip":"clip.mp4","prompt":"...","session":"out"}' | resolver tool   # agent contract
python -m resolver.mcp_server                   # same handler as MCP tools (needs the mcp extra)

resolver eval --conditions A,B,C --out eval/run # blind review pages per prompt per rater
resolver eval --score eval/run                  # after ratings are downloaded into eval/run/ratings/

resolver analyse clip.mp4 --out out             # speech regions and hit points only
resolver fit clip.mp4 track.mp3 --out out       # fit one track directly
```

Exit codes: 0 ok, 1 verification failed, 2 no match, 3 bad input, 4 dependency missing (the degraded result is still written).

`resolver fit` overrides:

| Flag | Use |
| --- | --- |
| `--hit 7.2` | Set the main hit, in seconds (always synced, even if detection found no clear hit) |
| `--speech 1.2-3.4,5-7.5` | Set the speech regions by hand |
| `--no-speech` | Turn off ducking |
| `--role underscore` | Mix role: `underscore` (music 12 LU below dialogue) or `featured` (4 LU). Default: from the amount of speech |
| `--music-lu 8` | Music loudness below dialogue, in LU, instead of the role's value |
| `--duck-db -14` | Minimum further reduction during speech. Each region is ducked deeper if needed for the 10 dB margin |
| `--config resolver.toml` | Tunable values; defaults in `resolver.example.toml` |

## Degraded modes

| Missing | What happens |
| --- | --- |
| `ANTHROPIC_API_KEY` or network | Offline brief from `src/resolver/data/brief_keywords.toml`, marked `mode: offline` (D1, D2) |
| Model returns bad JSON | One retry with the error, then the offline brief (D5) |
| CLAP (torch/transformers) | Tag-only index, BM25 over tags and titles (D3) |
| Beat This! / Silero | librosa beats, energy-band speech (D4) |
| Render fails | `report.json` with `status: render_failed` and the ffmpeg error (D6) |

## Output

| File | Contents |
| --- | --- |
| `session.json` | Brief, shown tracks, fallback state, refinement history |
| `brief.json`, `candidates.json` | The brief and the latest `ResolveResult` |
| `previews/1.mp4` ... | 360p preview of each candidate mixed under the clip |
| `review.html` | Brief, previews, reasons, licences, 1 to 5 ratings with a download button |
| `hit_marker.json` | Only when no music is placed (L9): where the in-app sound should hit |
| `clip.json` | Speech regions and ranked hit points |
| `mix.json` | Track offset, hit sync, loop, role, resolved levels, regions and warnings |
| `report.json` | Measured checks |
| `output.mp4` | Original video stream with the new mix |
| `stems/` | Original audio, placed music and mix |

## Checks

| Check | Target |
| --- | --- |
| Audio and video end times (and audio not starting after video) | Within 0.1 s |
| Music present | Above -60 dB |
| Peak level | At most 1.0 |
| Integrated loudness of the mix | -14 LUFS, within 1 LU |
| True peak of the mix (4x oversampled) | At most -1 dBTP |
| Clip audio above music, in speech regions | At least 10 dB |
| Hit to nearest beat, re-tracked on the placed music with the same engine | At most 80 ms. Skipped, with the reason, when there is no hit sync |
| Licence allowed by the profile at render time | `demo` or `commercial` |
| Attribution line present | Required for every CC BY variant |

## How the mix is set

1. Speech regions come from Silero VAD if installed, otherwise the energy-band detector.
2. The music is aligned so a downbeat (Beat This!) or beat (librosa) lands on the main hit. There is no hit sync when the top hit is weak (score below 0.35) or the track has no reliable pulse; the music then starts on a bar line. On clips under 3 s it opens on its strongest onset instead, with no fade-in.
3. A track too short for the clip loops whole bars, with a 30 ms equal-power crossfade at each joint.
4. The music is normalised to dialogue loudness minus the role offset, so the result does not depend on how loud the track was mastered.
5. Ducking reaches full depth as speech starts (80 ms attack) and recovers over 300 ms. Each region is ducked as far as needed for the 10 dB margin.
6. The mix is normalised to -14 LUFS and limited to -1 dBTP. The same gain goes on the stems, so they still sum to the mix. If limiting narrows the dialogue/music balance by more than 2 LU, a warning says so.

## Known limits

- **Embedding model.** The spec named `laion/larger_clap_music`. As published (checked 2026-11-09, with transformers 4.57 and 5.22), it gives every text prompt nearly the same embedding (cosine 0.999) and text-to-audio similarity of about 0.01. The default is `laion/clap-htsat-unfused`, which separates test sounds correctly. Switch back with `[embed] model` once a fixed checkpoint exists. Changing the model rebuilds the index.
- **Hit thresholds on real footage.** On the 12.9 s reel the strongest audio onset has z 4.9 (threshold 20) and the strongest motion change scores 0.30 (threshold 0.35), so fit does no hit sync. M4 labels decide whether the thresholds or the detection change. See `fixtures/real/NOTES.md`.
- **No-match threshold.** `no_match.z_threshold = 2.5` is a placeholder until `resolver eval --score` tunes it on the labelled half of the prompts.
- **Offline briefs** map known names and keywords only. Unknown references fall back to the words of the prompt, minus names.
- Without the `ml` extra, speech detection is the energy-band stand-in (it also fires on loud non-speech), and beats come from librosa with no tracked downbeats.
- The verifier checks the hit against the nearest re-tracked beat. It reports, but does not judge, whether that beat is also a re-tracked downbeat.
- On peaky dialogue, reaching -14 LUFS under -1 dBTP needs heavy limiting, which narrows the dialogue/music balance (reported as `final_balance_lu`).
- Loudness, peak and masking are measured on the stems before AAC encoding, not on the encoded file (`speech_masking_encoded` is not built).
- Not built: the ending on a downbeat (F5), detecting music already in the clip (C3), trying other segments of the same track before looping (T3), and a generation provider (interface only).
