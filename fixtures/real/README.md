# Real fixtures (not committed)

Put these here for the M4 real-clip test:

| File | Contents |
| --- | --- |
| `clip.mp4` | The original 12.9 s reel |
| `track.*` | One track you have the rights to use (CC0, CC BY, or royalty-free) |
| `labels.json` | Hand labels, format below |

```json
{
  "main_hit": 7.2,
  "speech_regions": [[0.4, 2.1], [3.0, 5.6]],
  "track_licence": "cc-by",
  "track_attribution": "Title by Artist, CC BY 4.0, <url>"
}
```

Label the hit by stepping through frames, and the speech regions by ear. Precision of 0.05 s is enough.
