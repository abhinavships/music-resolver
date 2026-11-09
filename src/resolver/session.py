"""Session on disk (6.2): resolve, refine, pick, skip, own, generate.

Everything a later call needs lives in `<out>/session.json`, so the CLI, the
agent tool and the review page can each pick a session up where it was left.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from . import brief as brief_mod
from . import clip as clip_stage
from . import fit as fit_stage
from . import media, retrieve as retrieve_mod, sfx as sfx_mod
from .config import Config
from .embed import Embedder
from .fallback import ChargeNotApproved, Fallback, Option
from .generation import GenerationProvider, make_provider
from .ingest import Index
from .library import attribution, licence_warnings
from .library.base import PROFILES
from .render import preview, render
from .schemas import (
    Brief, Candidate, Check, ClipAnalysis, Region, ResolveResult, TrackRecord, VerifyReport,
)
from .verify import verify


class FallbackState(BaseModel):
    current: str = "licensed"
    tried: list[str] = Field(default_factory=list)
    chosen: str | None = None
    charged: float = 0.0
    generation_cost: float | None = None

    @classmethod
    def of(cls, fb: Fallback) -> "FallbackState":
        return cls(current=fb.current.value, tried=list(fb.tried), chosen=fb.chosen.value if fb.chosen else None,
                   charged=fb.charged, generation_cost=fb.generation_cost)

    def restore(self) -> Fallback:
        fb = Fallback(generation_cost=self.generation_cost)
        fb.current, fb.tried, fb.charged = Option(self.current), list(self.tried), self.charged
        fb.chosen = Option(self.chosen) if self.chosen else None
        return fb


class SessionState(BaseModel):
    prompt: str
    clip_path: str
    reference: str | None = None
    brief: Brief
    clip: ClipAnalysis
    candidates: list[Candidate] = Field(default_factory=list)
    shown: list[str] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    fallback: FallbackState = Field(default_factory=FallbackState)
    last_result: ResolveResult | None = None


@dataclass
class Context:
    """What a session call needs besides the folder: config, index, models."""
    cfg: Config
    index: Index | None = None
    embedder: Embedder | None = None
    brief_call: brief_mod.ModelCall | None = None
    offline_brief: bool = False
    provider: GenerationProvider | None = None
    previews: bool = True


def _load(out: Path) -> SessionState:
    path = out / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"no session at {out} (run resolve first)")
    return SessionState.model_validate_json(path.read_text())


def _save(out: Path, state: SessionState) -> None:
    (out / "session.json").write_text(state.model_dump_json(indent=2))


def _write(path: Path, model) -> None:
    path.write_text(model.model_dump_json(indent=2))


def _message(status: str, candidates: list[Candidate], brief: Brief, fb: Fallback, skipped: bool = False) -> str:
    if status == "ok":
        head = f"{len(candidates)} licensed match{'es' if len(candidates) != 1 else ''} ready to preview"
        if brief.named_reference:
            head += f"; the original {brief.named_reference.kind} music is not available, so these match its sound"
        return head + "."
    lead = "Licensed matches skipped" if skipped else "No licensed track matched"
    if fb.current is Option.OWN_FILE:
        return f"{lead}; add your own file (with rights) or skip."
    if fb.current is Option.GENERATE:
        return f"{lead}; music can be generated for {fb.generation_cost} (needs approval)."
    return "No music placed; the clip was exported with a hit marker so you can add music in the posting app."


def _audio_vector(path: str, ctx: Context) -> np.ndarray | None:
    if ctx.embedder is None or ctx.index is None or ctx.index.tag_mode:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        y = media.load_audio(path, ctx.embedder.sr, Path(tmp))
    seg = int(ctx.cfg.index.segment_s * ctx.embedder.sr)
    windows = [y[s:s + seg] for s in range(0, max(1, len(y) - seg + 1), seg)][:12] or [y]
    return ctx.embedder.audio(windows).mean(axis=0)


def _segment_vector(c: Candidate, ctx: Context) -> np.ndarray | None:
    idx = ctx.index
    if idx is None or idx.vectors is None:
        return None
    rows = [i for i, (t, s) in enumerate(zip(idx.seg_tracks, idx.seg_starts))
            if t == c.track.track_id and abs(s - c.segment_start) < 1e-3]
    return idx.vectors[rows[0]] if rows else None


def _search(state: SessionState, ctx: Context, out: Path, exclude: set[str],
            extra: list[np.ndarray] | None, note: str) -> ResolveResult:
    cfg = ctx.cfg
    fb = state.fallback.restore()
    if state.brief.music_requested == "none":
        result = ResolveResult(status="ok", brief=state.brief, fallback_option=fb.current.value,
                               message="No music was requested; nothing to do.")
        state.last_result = result
        return result
    if ctx.index is None or ctx.index.size == 0:
        status, cands, searched, cov = "no_match", [], {"library_tracks": 0, "note": "no index built"}, []
    else:
        status, cands, searched, cov = retrieve_mod.retrieve(
            state.brief, ctx.index, ctx.embedder, state.clip, cfg, exclude=exclude, extra_vectors=extra)
    if status == "no_match" and fb.current is Option.LICENSED:
        fb.no_match(f"{sum(len(q) for q in searched.get('queries', {}).values())} queries over "
                    f"{searched.get('library_tracks', 0)} tracks")
        if fb.current is Option.NONE:
            _music_slot_export(state, out, cfg)
    if ctx.previews and cands:
        (out / "previews").mkdir(exist_ok=True)
        for c in cands:
            try:
                c.preview_path = str(_preview(state, c, out, cfg))
            except Exception as err:  # noqa: BLE001  (a failed preview never blocks the result)
                c.warnings.append(f"preview failed: {str(err)[-200:]}")
    state.candidates = cands
    state.shown = list(dict.fromkeys([*state.shown, *(c.track.track_id for c in cands)]))
    state.fallback = FallbackState.of(fb)
    state.history.append({"action": note, "status": status, "shown": [c.track.track_id for c in cands]})
    result = ResolveResult(
        status=status, brief=state.brief, candidates=cands, searched=searched,
        fallback_option=fb.current.value, coverage_warnings=cov, message=_message(status, cands, state.brief, fb),
        quoted_cost=fb.generation_cost if fb.current is Option.GENERATE else None,
    )
    if state.brief.music_requested == "implied" and status == "ok":
        result.message += " Music was implied, so nothing is applied until you pick one."
    state.last_result = result
    _write(out / "candidates.json", result)
    return result


def _preview(state: SessionState, c: Candidate, out: Path, cfg: Config) -> Path:
    work = out / "previews" / f"work{c.rank}"
    with tempfile.TemporaryDirectory() as tmp:
        y = media.load_audio(c.track.path, fit_stage.SR, Path(tmp))
    plan = fit_stage.plan_mix(state.clip, c.track.path, y, cfg=cfg, prefer_start=c.segment_start)
    _, artifacts = render(plan, work / "full.mp4", work, cfg)
    dest = out / "previews" / f"{c.rank}.mp4"
    preview(artifacts["video"], dest, cfg.session.preview_height)
    shutil.rmtree(work, ignore_errors=True)
    return dest


def _music_slot_export(state: SessionState, out: Path, cfg: Config) -> dict:
    """L9: the edit without music, plus where the music should hit, for the posting app's own sounds."""
    main = state.clip.hits[0] if state.clip.hits else None
    marker = {
        "hit_time": main.time if main else None,
        "hit_score": main.score if main else None,
        "next_hits": [h.time for h in state.clip.hits[1:3]],
        "speech_regions": [[r.start, r.end] for r in state.clip.speech_regions],
        "note": "Add the in-app sound and line its beat up with hit_time.",
    }
    (out / "hit_marker.json").write_text(json.dumps(marker, indent=2))
    media._run(["ffmpeg", "-y", "-v", "error", "-i", state.clip_path, "-c", "copy", str(out / "output.mp4")])
    fb = state.fallback.restore()
    report = VerifyReport(status="ok", passed=True, checks=[], skipped=[
        "no music placed (fallback reached 'none')", fb.summary(),
        f"hit marker at {marker['hit_time']} s written to hit_marker.json" if main else "no hit found for the marker",
    ])
    _write(out / "report.json", report)
    return marker


def resolve(clip_path: str, prompt: str, out: str | Path, ctx: Context, reference: str | None = None,
            analysis: ClipAnalysis | None = None) -> ResolveResult:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ctx.cfg
    analysis = analysis or clip_stage.analyse(clip_path, cfg)
    brief = brief_mod.make_brief(prompt, cfg, call=ctx.brief_call, offline=ctx.offline_brief)
    _write(out / "clip.json", analysis)
    _write(out / "brief.json", brief)
    provider = ctx.provider if ctx.provider is not None else make_provider(cfg)
    cost = provider.quote(brief, analysis.duration) if provider else None
    state = SessionState(prompt=prompt, clip_path=str(clip_path), reference=reference, brief=brief, clip=analysis,
                         fallback=FallbackState.of(Fallback(generation_cost=cost)))
    extra = None
    if reference:
        v = _audio_vector(reference, ctx)
        extra = [v] if v is not None else None
    result = _search(state, ctx, out, set(), extra, "resolve")
    _save(out, state)
    return result


def refine(out: str | Path, instruction: str, ctx: Context) -> ResolveResult:
    out = Path(out)
    state = _load(out)
    d = brief_mod.directives(instruction)
    exclude: set[str] = set() if d["include_shown"] else set(state.shown)
    extra = None
    if d["like_rank"]:
        cand = next((c for c in state.candidates if c.rank == d["like_rank"]), None)
        if cand is None:
            raise ValueError(f"no candidate #{d['like_rank']} in this session")
        v = _segment_vector(cand, ctx)
        extra = [v] if v is not None else None
        if extra is None:   # tag index: "like #2" means its tags
            state.brief = state.brief.model_copy(update={"primary": state.brief.primary.model_copy(
                update={"queries": [" ".join(cand.track.tags)] * 2})})
    elif d["different"]:
        scorer = retrieve_mod.Scorer(ctx.index, ctx.embedder)
        exclude |= retrieve_mod.neighbours(state.shown, scorer, ctx.cfg.session.different_neighbours)
    else:
        state.brief = brief_mod.refine(state.brief, instruction, ctx.cfg, call=ctx.brief_call)
    _write(out / "brief.json", state.brief)
    result = _search(state, ctx, out, exclude, extra, f"refine: {instruction}")
    _save(out, state)
    return result


def licence_checks(track: TrackRecord, profile: str) -> list[Check]:
    checks = [Check(name="licence_allowed", passed=track.licence in PROFILES[profile], target=f"allowed in '{profile}'",
                    detail=track.licence)]
    if track.licence.startswith("cc-by"):
        line = attribution(track)
        checks.append(Check(name="attribution_present", passed=bool(line), target="credit line for CC BY",
                            detail=line or "missing"))
    return checks


def _fit_and_report(state: SessionState, track: TrackRecord, out: Path, ctx: Context,
                    segment_start: float | None, overrides: dict) -> tuple[VerifyReport, dict]:
    cfg = ctx.cfg
    analysis = state.clip.model_copy(deep=True)
    if overrides.get("speech") is not None:
        analysis.speech_regions = [Region(start=float(a), end=float(b)) for a, b in overrides["speech"]]
    with tempfile.TemporaryDirectory() as tmp:
        y = media.load_audio(track.path, fit_stage.SR, Path(tmp))
    plan = fit_stage.plan_mix(analysis, track.path, y, hit_time=overrides.get("hit"), cfg=cfg,
                              role=overrides.get("role"), prefer_start=segment_start)
    plan.sfx = sfx_mod.plan_sfx(analysis, cfg)
    _write(out / "mix.json", plan)
    try:
        plan, artifacts = render(plan, out / "output.mp4", out / "stems", cfg)
    except RuntimeError as exc:
        report = VerifyReport(status="render_failed", passed=False, checks=[], skipped=[str(exc)[-1500:]])
        _write(out / "report.json", report)
        return report, {}
    _write(out / "mix.json", plan)
    report = verify(plan, artifacts, cfg)
    report.checks.extend(licence_checks(track, cfg.library.profile))
    report.passed = all(c.passed for c in report.checks)
    report.status = "ok" if report.passed else "verify_failed"
    report.attribution = attribution(track)
    report.licence_warnings = licence_warnings(track.licence)
    fb = state.fallback.restore()
    report.skipped.append(fb.summary())
    _write(out / "report.json", report)
    return report, {**artifacts, "report": str(out / "report.json"), "mix_plan": str(out / "mix.json")}


def pick(out: str | Path, rank: int, ctx: Context, overrides: dict | None = None) -> tuple[VerifyReport, dict]:
    out = Path(out)
    state = _load(out)
    cand = next((c for c in state.candidates if c.rank == rank), None)
    if cand is None:
        raise ValueError(f"no candidate #{rank}; shown: {[c.rank for c in state.candidates]}")
    fb = state.fallback.restore()
    if fb.current is not Option.LICENSED or fb.finished:
        raise ValueError(f"licensed option is no longer current (now {fb.current.value})")
    fb.pick()
    state.fallback = FallbackState.of(fb)
    state.history.append({"action": "pick", "rank": rank, "track": cand.track.track_id})
    _save(out, state)
    report, paths = _fit_and_report(state, cand.track, out, ctx, cand.segment_start, overrides or {})
    if not report.passed:
        # D7: the session still offers the other candidates.
        fb = state.fallback.restore()
        fb.chosen = None
        state.fallback = FallbackState.of(fb)
        _save(out, state)
    return report, paths


def skip(out: str | Path, ctx: Context, reason: str = "skipped by user") -> ResolveResult:
    out = Path(out)
    state = _load(out)
    fb = state.fallback.restore()
    if fb.finished:
        raise ValueError(f"fallback already finished ({fb.chosen.value})")
    fb.skip(reason)
    state.fallback = FallbackState.of(fb)
    state.history.append({"action": "skip", "reason": reason, "now": fb.current.value})
    if fb.current is Option.NONE:
        _music_slot_export(state, out, ctx.cfg)
    status = "needs_input" if fb.current in (Option.OWN_FILE, Option.GENERATE) else "no_match"
    result = ResolveResult(status=status, brief=state.brief, fallback_option=fb.current.value,
                           message=_message("skip", [], state.brief, fb, skipped=True),
                           quoted_cost=fb.generation_cost if fb.current is Option.GENERATE else None)
    state.last_result = result
    _save(out, state)
    return result


def own(out: str | Path, track_path: str, ctx: Context, rights_confirmed: bool,
        overrides: dict | None = None) -> tuple[VerifyReport, dict]:
    """L7: the user's own file, only after they confirm the rights."""
    out = Path(out)
    state = _load(out)
    if not rights_confirmed:
        raise PermissionError("confirm you have the rights to this file (--i-have-rights / rights_confirmed)")
    fb = state.fallback.restore()
    if fb.current is Option.LICENSED and not fb.finished:
        fb.skip("user chose their own file")
    if fb.current is not Option.OWN_FILE or fb.finished:
        raise ValueError(f"own file is not the current option (now {fb.current.value})")
    fb.pick()
    state.fallback = FallbackState.of(fb)
    state.history.append({"action": "own", "track": track_path, "rights_confirmed": True})
    _save(out, state)
    track = TrackRecord(track_id=f"own:{Path(track_path).name}", title=Path(track_path).stem, artist="user",
                        duration=media.duration_of(track_path), path=str(track_path), licence="own")
    report, paths = _fit_and_report(state, track, out, ctx, None, overrides or {})
    report.skipped.append("own file: user confirmed rights (L7)")
    _write(out / "report.json", report)
    return report, paths


def generate(out: str | Path, ctx: Context, approved_cost: float | None = None) -> ResolveResult | tuple:
    """Quote first; generate only when `approved_cost` equals the quote."""
    out = Path(out)
    state = _load(out)
    fb = state.fallback.restore()
    if fb.current is not Option.GENERATE:
        return ResolveResult(status="error", brief=state.brief, fallback_option=fb.current.value,
                             message=f"generation is not the current option (now {fb.current.value})")
    try:
        fb.pick(approved_cost=approved_cost)
    except ChargeNotApproved:
        return ResolveResult(status="needs_input", brief=state.brief, fallback_option=fb.current.value,
                             quoted_cost=fb.generation_cost,
                             message=f"Generating music costs {fb.generation_cost}; resend with approved_cost equal to it.")
    provider = ctx.provider or make_provider(ctx.cfg)
    path = provider.generate(state.brief, state.clip.duration, out / "generated.wav")
    state.fallback = FallbackState.of(fb)
    state.history.append({"action": "generate", "charged": approved_cost})
    _save(out, state)
    track = TrackRecord(track_id="generated:1", title="generated", artist=provider.name, duration=state.clip.duration,
                        path=str(path), licence="own")
    return _fit_and_report(state, track, out, ctx, None, {})
