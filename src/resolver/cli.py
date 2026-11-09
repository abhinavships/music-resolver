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


EXIT = {"ok": 0, "verify_failed": 1, "no_match": 2, "needs_input": 0, "error": 3, "render_failed": 1}


def _session_ctx(args: argparse.Namespace):
    from .app import make_context

    cfg = config_mod.load(args.config)
    ctx, notes = make_context(cfg, offline=getattr(args, "offline", False),
                              previews=not getattr(args, "no_previews", False))
    for n in notes:
        print(f"note: {n}", file=sys.stderr)
    return ctx, notes


def _print_result(result, out: Path) -> int:
    print(result.model_dump_json(indent=2))
    print(f"\n{result.message}", file=sys.stderr)
    for c in result.candidates:
        print(f"  #{c.rank} {c.track.title} - {c.track.artist} [{c.track.licence}] {c.reason}"
              f"{' | ' + '; '.join(c.warnings) if c.warnings else ''}", file=sys.stderr)
    if (out / "review.html").exists():
        print(f"review page: {out / 'review.html'}", file=sys.stderr)
    return EXIT.get(result.status, 0)


def _print_report(report, paths: dict) -> int:
    for check in report.checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name}: {check.value} (target {check.target}) {check.detail}")
    for note in report.skipped:
        print(f"SKIP  {note}")
    if report.attribution:
        print(f"credit: {report.attribution}")
    if paths.get("video"):
        print(f"output: {paths['video']}")
    return EXIT.get(report.status, 0)


def _overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    if getattr(args, "hit", None) is not None:
        out["hit"] = args.hit
    if getattr(args, "speech", None):
        out["speech"] = [tuple(map(float, item.split("-"))) for item in args.speech.split(",")]
    return out


def cmd_resolve(args: argparse.Namespace) -> int:
    from . import session

    ctx, notes = _session_ctx(args)
    out = Path(args.out)
    result = session.resolve(args.clip, args.prompt, out, ctx, reference=args.reference)
    result.coverage_warnings.extend(notes)
    _review(out)
    return _print_result(result, out)


def _review(out: Path) -> None:
    try:
        from .review import write_review
        write_review(out)
    except ImportError:
        pass


def cmd_refine(args: argparse.Namespace) -> int:
    from . import session

    ctx, _ = _session_ctx(args)
    result = session.refine(args.session, args.instruction, ctx)
    _review(Path(args.session))
    return _print_result(result, Path(args.session))


def cmd_pick(args: argparse.Namespace) -> int:
    from . import session

    ctx, _ = _session_ctx(args)
    report, paths = session.pick(args.session, args.rank, ctx, _overrides(args))
    return _print_report(report, paths)


def cmd_skip(args: argparse.Namespace) -> int:
    from . import session

    ctx, _ = _session_ctx(args)
    return _print_result(session.skip(args.session, ctx, args.reason), Path(args.session))


def cmd_own(args: argparse.Namespace) -> int:
    from . import session

    ctx, _ = _session_ctx(args)
    try:
        report, paths = session.own(args.session, args.track, ctx, rights_confirmed=args.i_have_rights,
                                    overrides=_overrides(args))
    except PermissionError as err:
        print(f"error: {err}", file=sys.stderr)
        return 3
    return _print_report(report, paths)


def cmd_eval(args: argparse.Namespace) -> int:
    from . import evaluate
    from .app import make_context
    from .schemas import Brief, CueSpec

    cfg = config_mod.load(args.config)
    prompts = evaluate.read_jsonl(args.prompts)
    if args.score:
        labels = {r["id"]: r["matchable"] for r in evaluate.read_jsonl(args.labels) if r.get("matchable") is not None}
        reports = [json.loads(p.read_text()) for p in sorted(Path(args.score).glob("picks/*/report.json"))]
        rep = evaluate.score(args.score, prompts, labels, cfg, reports)
        print(evaluate.markdown(rep))
        return 0
    ctx, notes = make_context(cfg, offline=args.offline, previews=False)
    for n in notes:
        print(f"note: {n}", file=sys.stderr)
    if ctx.index is None:
        return 4
    hand = {}
    for row in evaluate.read_jsonl(args.hand_briefs):
        b = row["brief"]
        hand[row["id"]] = Brief.model_validate(b) if "primary" in b else Brief(
            version="hand", mode="offline", primary=CueSpec.model_validate(b))
    conditions = tuple(c.strip() for c in args.conditions.split(","))
    run = evaluate.run(prompts, ctx.index, ctx.embedder, cfg, args.out, hand, conditions, args.raters,
                       offline=args.offline)
    missing = [p["id"] for p in prompts if "C" in conditions and p["id"] not in hand]
    if missing:
        print(f"note: condition C skipped for {len(missing)} prompts without a hand brief", file=sys.stderr)
    print(f"wrote {run.out}; rate the pages under {run.out / 'review'} then run "
          f"`resolver eval --score {run.out}`")
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

    def session_args(p, offline=True, previews=True):
        p.add_argument("--config", help="resolver.toml (default: ./resolver.toml if present)")
        if offline:
            p.add_argument("--offline", action="store_true", help="offline brief (keyword map, no model)")
        if previews:
            p.add_argument("--no-previews", action="store_true", help="skip rendering preview videos")

    resolve = sub.add_parser("resolve", help="brief + clip analysis + top 3 + previews + review page")
    resolve.add_argument("clip")
    resolve.add_argument("prompt")
    resolve.add_argument("--reference", help="reference audio file (searched by sound, never placed)")
    resolve.add_argument("--out", default="out")
    session_args(resolve)
    resolve.set_defaults(func=cmd_resolve)

    refine = sub.add_parser("refine", help='edit the brief and search again ("more comic", "like #2")')
    refine.add_argument("session")
    refine.add_argument("instruction")
    session_args(refine)
    refine.set_defaults(func=cmd_refine)

    pick = sub.add_parser("pick", help="fit, render and verify a candidate")
    pick.add_argument("session")
    pick.add_argument("rank", type=int)
    pick.add_argument("--hit", type=float)
    pick.add_argument("--speech")
    session_args(pick, offline=False, previews=False)
    pick.set_defaults(func=cmd_pick)

    skip = sub.add_parser("skip", help="move to the next fallback option")
    skip.add_argument("session")
    skip.add_argument("--reason", default="skipped by user")
    session_args(skip, offline=False, previews=False)
    skip.set_defaults(func=cmd_skip)

    own = sub.add_parser("own", help="use your own track (L7)")
    own.add_argument("session")
    own.add_argument("track")
    own.add_argument("--i-have-rights", action="store_true")
    own.add_argument("--hit", type=float)
    own.add_argument("--speech")
    session_args(own, offline=False, previews=False)
    own.set_defaults(func=cmd_own)

    ev = sub.add_parser("eval", help="run conditions A/B/C and write blind review pages, or --score them")
    ev.add_argument("--prompts", default="eval/prompts.jsonl")
    ev.add_argument("--conditions", default="A,B,C")
    ev.add_argument("--hand-briefs", default="eval/hand_briefs.jsonl")
    ev.add_argument("--labels", default="eval/labels.jsonl")
    ev.add_argument("--raters", type=int, default=3)
    ev.add_argument("--out", default="eval/run")
    ev.add_argument("--offline", action="store_true")
    ev.add_argument("--score", metavar="RUN_DIR", help="score a finished run from its ratings/ folder")
    ev.add_argument("--config", help="resolver.toml (default: ./resolver.toml if present)")
    ev.set_defaults(func=cmd_eval)

    tool = sub.add_parser("tool", help="agent contract: one JSON request on stdin, JSON on stdout")
    tool.set_defaults(func=lambda a: __import__("resolver.tool", fromlist=["main"]).main())

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
