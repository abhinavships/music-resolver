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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
