"""Command line: analyse a clip, or fit a chosen track to it."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import clip as clip_stage
from . import config as config_mod
from . import fit as fit_stage
from . import media
from .render import render
from .schemas import Region, VerifyReport
from .verify import verify


def _write(path: Path, model) -> None:
    path.write_text(model.model_dump_json(indent=2))


def cmd_analyse(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    analysis = clip_stage.analyse(args.clip, config_mod.load(args.config))
    _write(out / "clip.json", analysis)
    print(analysis.model_dump_json(indent=2))
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_mod.load(args.config)
    analysis = clip_stage.analyse(args.clip, cfg)
    if args.no_speech:
        analysis.speech_regions = []
    if args.speech:
        analysis.speech_regions = [
            Region(start=float(a), end=float(b))
            for a, b in (item.split("-") for item in args.speech.split(","))
        ]
    _write(out / "clip.json", analysis)

    with tempfile.TemporaryDirectory() as tmp:
        y = media.load_audio(args.track, fit_stage.SR, Path(tmp))
    plan = fit_stage.plan_mix(
        analysis, args.track, y, hit_time=args.hit, cfg=cfg, role=args.role,
        duck_db=args.duck_db, music_lu_below_dialogue=args.music_lu,
    )
    _write(out / "mix.json", plan)

    try:
        plan, artifacts = render(plan, out / "output.mp4", out / "stems", cfg)
    except RuntimeError as exc:
        # D6: keep the artifacts written so far and say why.
        report = VerifyReport(status="render_failed", passed=False, checks=[], skipped=[str(exc)[-1500:]])
        _write(out / "report.json", report)
        print(f"render failed: {exc}", file=sys.stderr)
        return 1
    _write(out / "mix.json", plan)
    report = verify(plan, artifacts, cfg)
    _write(out / "report.json", report)

    for warning in plan.warnings:
        print(f"warning: {warning}")
    for check in report.checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name}: {check.value} (target {check.target}) {check.detail}")
    for note in report.skipped:
        print(f"SKIP  {note}")
    print(f"output: {out / 'output.mp4'}")
    return 0 if report.passed else 1


def cmd_index(args: argparse.Namespace) -> int:
    from .embed import make_embedder
    from .ingest import ingest
    from .library import make_adapters

    cfg = config_mod.load(args.config)
    embedder = None if args.tags_only else make_embedder(cfg.embed.model, cfg.embed.batch_size, cfg.library.cache_dir)
    if embedder is None and not args.tags_only:
        print(f"warning: {cfg.embed.model} unavailable; building a tag-only index (D3)", file=sys.stderr)
    index = ingest(cfg, make_adapters(cfg), embedder)
    thin = [k for k, v in index.coverage["instruments"].items() if v["segments"] < cfg.retrieve.coverage_min_segments]
    if thin:
        print(f"coverage: fewer than {cfg.retrieve.coverage_min_segments} segments for {', '.join(thin)}")
    print(f"wrote {Path(cfg.index.dir) / 'coverage.json'}")
    return 4 if embedder is None and not args.tags_only else 0


def cmd_brief(args: argparse.Namespace) -> int:
    from .brief import make_brief

    cfg = config_mod.load(args.config)
    b = make_brief(args.prompt, cfg, offline=args.offline, use_cache=not args.no_cache)
    print(b.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resolver")
    sub = parser.add_subparsers(dest="command", required=True)

    analyse = sub.add_parser("analyse", help="find speech regions and hit points")
    analyse.add_argument("clip")
    analyse.add_argument("--out", default="out")
    analyse.add_argument("--config", help="resolver.toml (default: ./resolver.toml if present)")
    analyse.set_defaults(func=cmd_analyse)

    fit = sub.add_parser("fit", help="fit a licensed track to the clip and verify the render")
    fit.add_argument("clip")
    fit.add_argument("track")
    fit.add_argument("--out", default="out")
    fit.add_argument("--hit", type=float, help="override the detected main hit, in seconds")
    fit.add_argument("--speech", help="override speech regions, e.g. 1.2-3.4,5-7.5")
    fit.add_argument("--no-speech", action="store_true", help="treat the clip as having no speech")
    fit.add_argument("--role", choices=["underscore", "featured"], help="mix role (default: from speech amount)")
    fit.add_argument("--music-lu", type=float, help="music loudness below dialogue, in LU (default: from role)")
    fit.add_argument("--duck-db", type=float, help="minimum extra reduction during speech, in dB (default: config)")
    fit.add_argument("--config", help="resolver.toml (default: ./resolver.toml if present)")
    fit.set_defaults(func=cmd_fit)

    index = sub.add_parser("index", help="ingest library adapters, build the index, write coverage.json")
    index.add_argument("--config", help="resolver.toml (default: ./resolver.toml if present)")
    index.add_argument("--tags-only", action="store_true", help="skip CLAP; tag search only (D3)")
    index.set_defaults(func=cmd_index)

    brief = sub.add_parser("brief", help="turn a prompt into a music brief (prints brief.json)")
    brief.add_argument("prompt")
    brief.add_argument("--offline", action="store_true", help="keyword map only, no model call")
    brief.add_argument("--no-cache", action="store_true")
    brief.add_argument("--config", help="resolver.toml (default: ./resolver.toml if present)")
    brief.set_defaults(func=cmd_brief)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
