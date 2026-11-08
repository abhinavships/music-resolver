# music-resolver

Prototype of a reference-to-music resolver for video edits (Cardboard hard problem 06, applied to music).

## Read first

1. `SPEC.md`: the build spec. Section 0 has the rules, section 2 every scenario with an ID, section 8 the milestones and the tests each one needs.
2. `README.md`: what runs today.

## Working rules

- Build milestones in order (M4 next). End each milestone with `pytest` green and its acceptance tests in place.
- Name tests after scenario IDs, e.g. `test_l5_nd_always_excluded`.
- Never fetch or place copyrighted audio a user names; reference names are text only.
- Licence filtering happens before ranking. Unknown licence means excluded.
- Tunable numbers go in `resolver.example.toml` and are read through `config.py`.
- Tests that need network, API keys, models or real media are marked `live` or skip cleanly.
- Keep the user's video local; only prompt text goes to the LLM.

## Commands

```bash
pip install -e ".[dev]"
pytest                 # offline suite (uses Beat This!/Silero if the ml extra is installed, else fallbacks)
pytest -m live         # needs keys, models, network
resolver fit clip.mp4 track.mp3 --out out
```

## Layout

- `src/resolver/`: pipeline stages, one module per stage, JSON artifacts in `schemas.py`
- `tests/`: synthetic fixtures in `conftest.py`
- `eval/prompts.jsonl`: the 30 evaluation prompts
- `fixtures/real/`: real clip and track (not committed), see its README
