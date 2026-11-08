# music-resolver

Turns a music reference in an editing prompt into a licensed track fitted to the clip.

**Status:** milestones M1 and M2 are built: the fallback logic and the fit, mix and verify path, tested on synthetic media only. Brief generation, the music library and retrieval are specified in `SPEC.md` (M3 to M9) and not built yet.

## Run

```bash
pip install -e ".[dev]"
pytest

# Find speech regions and hit points
resolver analyse clip.mp4 --out out

# Fit a track you have the rights to, render it, and check the result
resolver fit clip.mp4 track.mp3 --out out
```

`resolver fit` exits with code 1 if a check fails. Useful overrides:

| Flag | Use |
| --- | --- |
| `--hit 7.2` | Set the main hit, in seconds |
| `--speech 1.2-3.4,5-7.5` | Set the speech regions by hand |
| `--no-speech` | Turn off ducking |
| `--music-db -8` | Music level |
| `--duck-db -14` | Further reduction during speech, relative to the music level |

## Output

| File | Contents |
| --- | --- |
| `clip.json` | Speech regions and ranked hit points |
| `mix.json` | Track offset, levels, regions and warnings |
| `report.json` | Measured checks |
| `output.mp4` | Original video stream with the new mix |
| `stems/` | Original audio, placed music and mix |

## Checks

| Check | Target |
| --- | --- |
| Audio and video duration difference | At most 0.1 s |
| Music present | Above -60 dB |
| Peak level | At most 1.0 |
| Clip audio above music, in speech regions | At least 10 dB |
| Hit to nearest beat, re-tracked on the placed music | At most 80 ms |

## Known limits

- Speech detection is an energy-band stand-in and also fires on loud non-speech.
- Beat tracking finds beats, not downbeats, so the hit may land on a weak bar position.
- The main hit comes from audio onsets and local motion. It has not been tested on real footage.
- Masking is measured on the stems before AAC encoding, not on the encoded file.
