"""Evaluation (M9): conditions A (raw prompt), B (generated brief), C (hand brief).

Two phases, because people rate in between:

1. `run(...)` retrieves for every prompt and condition, writes per-condition
   results and one blind review page per prompt per rater (labels hidden,
   order shuffled), plus the key that maps slots back to conditions.
2. `score(...)` reads the downloaded ratings, the matchable labels and any pick
   reports, and writes `report.md` / `report.json` with the pass bars.

The no-match threshold is tuned on half the prompts and reported on the other half.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import brief as brief_mod
from .config import Config
from .embed import Embedder
from .ingest import Index
from .retrieve import retrieve
from .review import write_blind_review
from .schemas import Brief, CueSpec, ResolveResult

CONDITIONS = ("A", "B", "C")
USABLE = 4            # a rating of 4 or 5 means "I would use this"


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def raw_brief(prompt: str, cfg: Config) -> Brief:
    """Condition A: the raw prompt is the only query."""
    return Brief(version="raw", mode="offline", primary=CueSpec(label="raw prompt", queries=[prompt]))


def split(ids: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic half split by id hash: (tune, report)."""
    h = {i: int(hashlib.sha256(i.encode()).hexdigest(), 16) for i in ids}
    ordered = sorted(ids, key=h.get)
    return ordered[::2], ordered[1::2]


@dataclass
class Run:
    out: Path
    results: dict[str, dict[str, dict]] = field(default_factory=dict)   # condition -> prompt id -> result


def run(prompts: list[dict], index: Index, embedder: Embedder | None, cfg: Config, out: str | Path,
        hand_briefs: dict[str, Brief] | None = None, conditions: tuple[str, ...] = CONDITIONS,
        raters: int = 3, brief_call=None, offline: bool = False) -> Run:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hand_briefs = hand_briefs or {}
    r = Run(out)
    for cond in conditions:
        r.results[cond] = {}
    for p in prompts:
        pid, prompt = p["id"], p["prompt"]
        pooled: list[tuple[str, ResolveResult]] = []
        for cond in conditions:
            if cond == "A":
                brief = raw_brief(prompt, cfg)
            elif cond == "B":
                brief = brief_mod.make_brief(prompt, cfg, call=brief_call, offline=offline)
            else:
                if pid not in hand_briefs:
                    continue
                brief = hand_briefs[pid]
            status, cands, searched, cov = retrieve(brief, index, embedder, None, cfg)
            res = ResolveResult(status=status, brief=brief, candidates=cands, searched=searched,
                                coverage_warnings=cov, message="")
            r.results[cond][pid] = res.model_dump(mode="json")
            pooled.append((cond, res))
        pdir = out / "review" / pid
        pdir.mkdir(parents=True, exist_ok=True)
        for k in range(1, raters + 1):
            write_blind_review(pooled, pdir, prompt, rater_seed=k)
    for cond, res in r.results.items():
        (out / f"results_{cond}.json").write_text(json.dumps(res, indent=2))
    (out / "README.md").write_text(
        "# Rating\n\nOpen `review/<prompt id>/review-rater<N>.html` (N = your rater number), rate every option "
        "1 to 5 (4 or more means you would use it), and press *Download ratings*. Put the downloaded files in "
        "`ratings/`. Then run `resolver eval --score <this folder>`.\n")
    return r


# --- agreement -----------------------------------------------------------------------------

def krippendorff_alpha(data: list[list[float | None]], level: str = "interval") -> float:
    """Krippendorff's alpha. `data[coder][unit]`, None for missing. Levels: nominal, ordinal, interval."""
    units = list(zip(*data))
    values = sorted({v for row in data for v in row if v is not None})
    if len(values) < 2:
        return 1.0
    pos = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for unit in units:
        vals = [v for v in unit if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i != j:
                    o[pos[a], pos[b]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")

    def delta(i: int, j: int) -> float:
        if level == "nominal":
            return 0.0 if i == j else 1.0
        if level == "interval":
            return (values[i] - values[j]) ** 2
        if level == "ordinal":
            lo, hi = min(i, j), max(i, j)
            return (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
        raise ValueError(level)

    k = len(values)
    d = np.array([[delta(i, j) for j in range(k)] for i in range(k)])
    observed = (o * d).sum()
    expected = (np.outer(n_c, n_c) * d).sum() / (n - 1)
    return float(1 - observed / expected) if expected > 0 else 1.0


# --- scoring ---------------------------------------------------------------------------------

def load_ratings(folder: Path, review_root: Path) -> list[dict]:
    """Flatten downloaded rating files to rows: prompt id, rater, condition, track, rating."""
    rows = []
    for f in sorted(Path(folder).glob("*.json")):
        data = json.loads(f.read_text())
        pid = data["session"]
        seed = data.get("rater_seed")
        key_path = review_root / pid / f"key-rater{seed}.json"
        key = json.loads(key_path.read_text()) if key_path.exists() else {}
        for r in data["ratings"]:
            if r["rating"] is None:
                continue
            rows.append({"prompt": pid, "rater": data.get("rater") or f"rater{seed}", "slot": r["slot"],
                         "condition": key.get(r["slot"]), "track_id": r["track_id"], "rating": r["rating"]})
    return rows


def usable_rate(rows: list[dict], condition: str, prompt_ids: list[str]) -> tuple[float | None, int]:
    """Share of prompts where at least one top-3 track has a median rating >= USABLE."""
    hits, n = 0, 0
    for pid in prompt_ids:
        per_track: dict[str, list[int]] = {}
        for r in rows:
            if r["prompt"] == pid and r["condition"] == condition:
                per_track.setdefault(r["track_id"], []).append(r["rating"])
        if not per_track:
            continue
        n += 1
        if any(np.median(v) >= USABLE for v in per_track.values()):
            hits += 1
    return (hits / n if n else None), n


def best_z(result: dict) -> float:
    zs = result.get("searched", {}).get("z", {})
    return max(zs.values()) if zs else 0.0


def tune_threshold(z: dict[str, float], matchable: dict[str, bool], ids: list[str],
                   default: float = 2.5) -> float:
    """z threshold that maximises no-match accuracy on `ids`.

    Candidates are midpoints between neighbouring z values of the tuning half,
    so the threshold does not sit exactly on a tuning prompt's score.
    """
    ids = [i for i in ids if i in z and i in matchable]
    if not ids:
        return default
    vals = sorted({z[i] for i in ids})
    cands = [vals[0] - 0.5, *[(a + b) / 2 for a, b in zip(vals, vals[1:])], vals[-1] + 0.5]
    best, best_acc = default, -1.0
    for t in cands:
        acc = float(np.mean([(z[i] >= t) == matchable[i] for i in ids]))
        if acc > best_acc + 1e-9:
            best, best_acc = round(t, 3), acc
    return best


def no_match_metrics(z: dict[str, float], matchable: dict[str, bool], ids: list[str], threshold: float) -> dict:
    ids = [i for i in ids if i in z and i in matchable]
    if not ids:
        return {"n": 0, "accuracy": None, "false_no_match": None}
    pred_match = {i: z[i] >= threshold for i in ids}
    acc = float(np.mean([pred_match[i] == matchable[i] for i in ids]))
    m = [i for i in ids if matchable[i]]
    fnm = float(np.mean([not pred_match[i] for i in m])) if m else None
    return {"n": len(ids), "accuracy": acc, "false_no_match": fnm, "threshold": threshold}


def pick_metrics(reports: list[dict], cfg: Config) -> dict:
    offsets, margins, n_sync = [], [], 0
    for rep in reports:
        checks = {c["name"]: c for c in rep.get("checks", [])}
        if "hit_offset" in checks and checks["hit_offset"].get("value") is not None:
            n_sync += 1
            offsets.append(checks["hit_offset"]["value"])
        if "speech_masking" in checks and checks["speech_masking"].get("value") is not None:
            margins.append(checks["speech_masking"]["value"])
    within = float(np.mean([o <= cfg.verify.max_hit_offset_s for o in offsets])) if offsets else None
    return {"clips": len(reports), "hit_synced": n_sync, "hit_within_80ms": within,
            "speech_margin_min_db": min(margins) if margins else None,
            "speech_margin_ok": all(m >= cfg.mix.min_speech_margin_db for m in margins) if margins else None}


def dead_ends(results: dict[str, dict[str, dict]]) -> list[str]:
    """A dead end: no candidates and no stated next step."""
    out = []
    for cond, per in results.items():
        for pid, res in per.items():
            if not res["candidates"] and res["status"] != "no_match":
                out.append(f"{cond}:{pid}")
    return out


def score(run_dir: str | Path, prompts: list[dict], labels: dict[str, bool], cfg: Config,
          reports: list[dict] | None = None) -> dict:
    run_dir = Path(run_dir)
    results = {c: json.loads((run_dir / f"results_{c}.json").read_text())
               for c in CONDITIONS if (run_dir / f"results_{c}.json").exists()}
    rows = load_ratings(run_dir / "ratings", run_dir / "review")
    ids = [p["id"] for p in prompts]
    matchable_ids = [i for i in ids if labels.get(i, True)]

    table = {}
    for cond in results:
        rate, n = usable_rate(rows, cond, matchable_ids)
        table[cond] = {"top3_usable": rate, "prompts_rated": n, "ratings": sum(r["condition"] == cond for r in rows)}

    # Agreement over (prompt, slot) units, raters as coders.
    raters = sorted({r["rater"] for r in rows})
    units = sorted({(r["prompt"], r["track_id"], r["condition"]) for r in rows})
    matrix = [[next((x["rating"] for x in rows if x["rater"] == rt and (x["prompt"], x["track_id"], x["condition"]) == u),
                    None) for u in units] for rt in raters]
    alpha = krippendorff_alpha(matrix, "ordinal") if len(raters) >= 2 and units else None

    tune_ids, report_ids = split(ids)
    nm = {}
    for cond, per in results.items():
        z = {pid: best_z(res) for pid, res in per.items()}
        t = tune_threshold(z, labels, tune_ids) if labels else cfg.no_match.z_threshold
        nm[cond] = {"tuned_on": len(tune_ids), **no_match_metrics(z, labels, report_ids, t)}

    a, b = table.get("A", {}).get("top3_usable"), table.get("B", {}).get("top3_usable")
    picks = pick_metrics(reports or [], cfg)
    bars = {
        "B >= A + 20 points": None if a is None or b is None else b >= a + 0.20,
        "no-match accuracy >= 80%": None if nm.get("B", {}).get("accuracy") is None else nm["B"]["accuracy"] >= 0.8,
        "false no-match <= 15%": None if nm.get("B", {}).get("false_no_match") is None else nm["B"]["false_no_match"] <= 0.15,
        "hit offset <= 80 ms on 90% of clips": None if picks["hit_within_80ms"] is None else picks["hit_within_80ms"] >= 0.9,
        "speech margin >= 10 dB": picks["speech_margin_ok"],
        "zero dead ends": not dead_ends(results),
    }
    failures = []
    for cond in ("B",):
        for pid, res in results.get(cond, {}).items():
            per_track: dict[str, list[int]] = {}
            for r in rows:
                if r["prompt"] == pid and r["condition"] == cond:
                    per_track.setdefault(r["track_id"], []).append(r["rating"])
            usable = any(np.median(v) >= USABLE for v in per_track.values())
            if labels.get(pid, True) and per_track and not usable:
                failures.append({"prompt": pid, "brief": res["brief"]["primary"],
                                 "top": [(c["track"]["title"], c["reason"]) for c in res["candidates"]]})
    report = {"sample": {"prompts": len(ids), "matchable": len(matchable_ids), "raters": len(raters),
                         "ratings": len(rows)},
              "conditions": table, "krippendorff_alpha_ordinal": alpha, "no_match": nm, "picks": picks,
              "pass_bars": bars, "dead_ends": dead_ends(results), "failures": failures, "raw_ratings": rows}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "report.md").write_text(markdown(report))
    return report


def _fmt(x, pct=False) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "pass" if x else "FAIL"
    return f"{x * 100:.0f}%" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def markdown(rep: dict) -> str:
    s = rep["sample"]
    lines = ["# Evaluation report", "",
             f"Prompts: {s['prompts']} ({s['matchable']} matchable). Raters: {s['raters']}. Ratings: {s['ratings']}.", "",
             "| Condition | Top-3 usable | Prompts rated | No-match accuracy | False no-match | z threshold |",
             "| --- | --- | --- | --- | --- | --- |"]
    for cond, row in rep["conditions"].items():
        nm = rep["no_match"].get(cond, {})
        lines.append(f"| {cond} | {_fmt(row['top3_usable'], True)} | {row['prompts_rated']} | "
                     f"{_fmt(nm.get('accuracy'), True)} | {_fmt(nm.get('false_no_match'), True)} | {_fmt(nm.get('threshold'))} |")
    lines += ["", f"Rater agreement (Krippendorff's alpha, ordinal): {_fmt(rep['krippendorff_alpha_ordinal'])}", "",
              "No-match threshold tuned on half the prompts, accuracy reported on the other half.", "",
              "## Pass bars", "", "| Bar | Result |", "| --- | --- |"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in rep["pass_bars"].items()]
    p = rep["picks"]
    lines += ["", f"Picks: {p['clips']} clips, {p['hit_synced']} hit-synced, within 80 ms: {_fmt(p['hit_within_80ms'], True)}, "
              f"minimum speech margin: {_fmt(p['speech_margin_min_db'])} dB.", "", "## Failures (condition B)", ""]
    for f in rep["failures"] or []:
        lines.append(f"- **{f['prompt']}**: brief `{f['brief']['label']}` ({', '.join(f['brief']['queries'][:2])}); "
                     f"top: {'; '.join(t for t, _ in f['top'])}")
    if not rep["failures"]:
        lines.append("None.")
    return "\n".join(lines) + "\n"
