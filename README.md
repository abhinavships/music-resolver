# music-resolver

Turns a music reference in an editing prompt into a licensed track fitted to the clip.

**Status:** milestones M1 to M3 are built: the fallback logic, and the analyse, fit, mix and verify path, hardened for variable frame rate, stream offsets, loudness, looping and missing engines. Everything is tested on synthetic media only. The real-clip test (M4), brief generation, the music library and retrieval are specified in `SPEC.md` (M4 to M9) and not built yet.

## Run

```bash
pip install -e ".[dev]"          # core: librosa beats, energy-band speech detection
pip install -e ".[dev,ml]"       # optional: Beat This! downbeats, Silero VAD (downloads torch)
pytest

# Find speech regions and hit points
resolver analyse clip.mp4 --out out

# Fit a track you have the rights to, render it, and check the result
resolver fit clip.mp4 track.mp3 --out out
```

`resolver fit` exits with code 1 if a check fails. Useful overrides:

| Flag | Use |
| --- | --- |
| `--hit 7.2` | Set the main hit, in seconds (always synced, even if detection found no clear hit) |
| `--speech 1.2-3.4,5-7.5` | Set the speech regions by hand |
| `--no-speech` | Turn off ducking |
| `--role underscore` | Mix role: `underscore` (music 12 LU below dialogue) or `featured` (4 LU). Default: from the amount of speech |
| `--music-lu 8` | Music loudness below dialogue, in LU, instead of the role's value |
| `--duck-db -14` | Minimum further reduction during speech. Each region is ducked deeper if needed for the 10 dB margin |
| `--config resolver.toml` | Tunable values; defaults in `resolver.example.toml` |

## Output

| File | Contents |
| --- | --- |
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

## How the mix is set

1. Speech regions come from Silero VAD if installed, otherwise the energy-band detector.
2. The music is aligned so a downbeat (Beat This!) or beat (librosa) lands on the main hit. There is no hit sync when the top hit is weak (score below 0.35) or the track has no reliable pulse; the music then starts on a bar line. On clips under 3 s it opens on its strongest onset instead, with no fade-in.
3. A track too short for the clip loops whole bars, with a 30 ms equal-power crossfade at each joint.
4. The music is normalised to dialogue loudness minus the role offset, so the result does not depend on how loud the track was mastered.
5. Ducking reaches full depth as speech starts (80 ms attack) and recovers over 300 ms. Each region is ducked as far as needed for the 10 dB margin.
6. The mix is normalised to -14 LUFS and limited to -1 dBTP. The same gain goes on the stems, so they still sum to the mix. If limiting narrows the dialogue/music balance by more than 2 LU, a warning says so.

## Known limits

- Without the `ml` extra, speech detection is the energy-band stand-in (it also fires on loud non-speech), and beats come from librosa with no tracked downbeats.
- The main hit comes from audio onsets and local motion. Its salience thresholds were calibrated on synthetic media only; M4 checks them on a real clip.
- The verifier checks the hit against the nearest re-tracked beat. It reports, but does not judge, whether that beat is also a re-tracked downbeat, because re-tracking can pick a different bar phase on ambiguous music.
- On peaky dialogue, reaching -14 LUFS under -1 dBTP needs heavy limiting, which narrows the dialogue/music balance (reported as `final_balance_lu`).
- Loudness, peak and masking are measured on the stems before AAC encoding, not on the encoded file.
- Not built yet: the ending on a downbeat (F5), detecting music already in the clip (C3), and the effects layer.
