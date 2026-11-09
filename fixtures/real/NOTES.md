# Real clip notes (M4)

## First automatic pass (2026-11-09, before hand labels)

Clip: `clip.mp4`, 12.87 s, 30 fps, from the WhatsApp export of 2026-10-30.

| Measure | Value | What it means |
| --- | --- | --- |
| Speech (Silero) | one region, 0.70 to 12.86 s | Almost the whole clip reads as voice. Check by ear whether that is speech, or music/crowd in the source (C3). |
| Audio onset peak | robust z 4.9 (at 12.07 s) | Below `hits.salience_z_low = 20`, so audio adds nothing to any hit. No clear sound event. |
| Motion peak | robust z 44 (at 12.17 s) | Scores 0.30, below `hits.min_score = 0.35`, so fit would not hit-sync (C7). |
| Top hits | 12.18 s (0.30), 0.04 s (0.24), 12.68 s (0.21) | Motion-only. 12.2 s may be a cut near the end. |

Open questions for the hand labels:

1. Where is the real main hit? If it is around 12.2 s, `hits.min_score` (or the 50/50 audio/motion weighting, which caps a motion-only hit at 0.5) is too strict for real footage.
2. Is 0.7 to 12.86 s really all speech? If music or crowd is in the source, C3 (source music detection, not built) matters for this reel.

## Listening note (fill in after `resolver fit fixtures/real/clip.mp4 fixtures/real/track.mp3 --out out/real`)

- Music level under speech:
- Does the beat land on the hit:
- Loop or ending artefacts:
- Overall: usable / not usable, and why:
