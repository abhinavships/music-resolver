"""M9: evaluation harness: conditions, blind pages, ratings, agreement, threshold split, report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from resolver import evaluate
from resolver.config import Config
from resolver.ingest import ingest
from resolver.library import LocalFolder
from resolver.schemas import Brief, CueSpec

from toy import ToyEmbedder, build_toy_library

ROOT = Path(__file__).resolve().parent.parent


def test_eval_prompts_file():
    prompts = evaluate.read_jsonl(ROOT / "eval" / "prompts.jsonl")
    assert len(prompts) == 30 and len({p["id"] for p in prompts}) == 30
    labels = evaluate.read_jsonl(ROOT / "eval" / "labels.template.jsonl")
    assert [l["id"] for l in labels] == [p["id"] for p in prompts]


def test_krippendorff_alpha_reference_values():
    # Krippendorff (2011), "Computing Krippendorff's alpha-reliability", 4 coders x 12 units.
    n = None
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
        [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n],
    ]
    assert evaluate.krippendorff_alpha(data, "nominal") == pytest.approx(0.743, abs=0.001)
    assert evaluate.krippendorff_alpha(data, "interval") == pytest.approx(0.849, abs=0.001)
    assert evaluate.krippendorff_alpha([[1, 2, 3], [1, 2, 3]], "ordinal") == pytest.approx(1.0)


def test_split_is_deterministic_halves():
    ids = [f"p{i}" for i in range(30)]
    tune, report = evaluate.split(ids)
    assert len(tune) == len(report) == 15 and not set(tune) & set(report)
    assert evaluate.split(list(reversed(ids))) == (tune, report)


def test_threshold_tuned_on_one_half_reported_on_other():
    z = {"a": 5.0, "b": 4.0, "c": 0.5, "d": 0.2, "e": 3.5, "f": 1.0}
    m = {"a": True, "b": True, "c": False, "d": False, "e": True, "f": False}
    t = evaluate.tune_threshold(z, m, ["a", "c", "e"])
    assert 0.5 < t <= 3.5
    rep = evaluate.no_match_metrics(z, m, ["b", "d", "f"], t)
    assert rep["accuracy"] == 1.0 and rep["false_no_match"] == 0.0


def test_usable_rate_uses_median_of_raters():
    rows = [{"prompt": "p1", "condition": "A", "track_id": "t1", "rating": r} for r in (5, 4, 1)] + \
           [{"prompt": "p2", "condition": "A", "track_id": "t2", "rating": r} for r in (3, 4, 2)]
    assert evaluate.usable_rate(rows, "A", ["p1", "p2"]) == (0.5, 2)


def test_pick_metrics():
    reports = [{"checks": [{"name": "hit_offset", "value": 0.02}, {"name": "speech_masking", "value": 14.0}]},
               {"checks": [{"name": "hit_offset", "value": 0.12}, {"name": "speech_masking", "value": 11.0}]}]
    m = evaluate.pick_metrics(reports, Config())
    assert m["hit_within_80ms"] == 0.5 and m["speech_margin_min_db"] == 11.0 and m["speech_margin_ok"]


@pytest.fixture(scope="module")
def toy_index(tmp_path_factory):
    root = tmp_path_factory.mktemp("m9")
    lib = build_toy_library(root / "lib", duration=30.0)
    cfg = Config()
    cfg.library.local_folder, cfg.index.dir, cfg.library.cache_dir = str(lib), str(root / "index"), str(root / "c")
    cfg.no_match.z_threshold = 1.5
    emb = ToyEmbedder()
    return cfg, ingest(cfg, [LocalFolder(lib)], emb, log=lambda *_: None), emb


PROMPTS = [
    {"id": "p1", "prompt": "comic dhol fight music"},
    {"id": "p2", "prompt": "calm ambient synth for a sunset"},
    {"id": "p3", "prompt": "carnatic violin over a chiptune beat"},
    {"id": "p4", "prompt": "suspense orchestral music when the door opens"},
]


def test_eval_end_to_end(toy_index, tmp_path):
    cfg, index, emb = toy_index
    hand = {"p1": Brief(version="hand", mode="offline", primary=CueSpec(
        label="hand", queries=["comic dhol tabla fight", "fast dhol percussion comic"]))}
    run = evaluate.run(PROMPTS, index, emb, cfg, tmp_path, hand, raters=3, offline=True)
    assert set(run.results["A"]) == set(run.results["B"]) == {"p1", "p2", "p3", "p4"}
    assert set(run.results["C"]) == {"p1"}                    # C only where a hand brief exists
    assert run.results["B"]["p3"]["status"] == "no_match"
    pages = sorted((tmp_path / "review" / "p1").glob("review-rater*.html"))
    assert len(pages) == 3 and "Dhol" not in pages[0].read_text()

    # Simulated raters: like condition B's picks, dislike condition A's.
    (tmp_path / "ratings").mkdir()
    for pid in ("p1", "p2", "p4"):
        for seed in (1, 2, 3):
            key = json.loads((tmp_path / "review" / pid / f"key-rater{seed}.json").read_text())
            meta = json.loads(pages[0].read_text().split('id="meta">')[1].split("</script>")[0]) if pid == "p1" else None
            page = (tmp_path / "review" / pid / f"review-rater{seed}.html").read_text()
            slots = json.loads(page.split('id="meta">')[1].split("</script>")[0])["slots"]
            ratings = [{"slot": s, "track_id": t, "rating": 5 if key[s] in ("B", "C") else 2} for s, t in slots.items()]
            (tmp_path / "ratings" / f"{pid}-{seed}.json").write_text(json.dumps(
                {"session": pid, "rater": f"r{seed}", "rater_seed": seed, "ratings": ratings}))
            assert meta is None or meta["rater_seed"] == 1

    labels = {"p1": True, "p2": True, "p3": False, "p4": True}
    rep = evaluate.score(tmp_path, PROMPTS, labels, cfg, reports=[])
    assert rep["conditions"]["B"]["top3_usable"] == 1.0
    assert rep["conditions"]["A"]["top3_usable"] < 1.0 or rep["conditions"]["A"]["prompts_rated"] == 0
    assert rep["pass_bars"]["zero dead ends"] is True
    assert rep["krippendorff_alpha_ordinal"] == pytest.approx(1.0)
    assert rep["sample"]["raters"] == 3 and rep["raw_ratings"]
    md = (tmp_path / "report.md").read_text()
    assert "| B |" in md and "Krippendorff" in md and "tuned on half" in md


def test_eval_cli_runs_offline(toy_index, tmp_path, monkeypatch):
    from resolver import cli
    cfg, index, emb = toy_index
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps(p) for p in PROMPTS[:2]) + "\n")
    monkeypatch.setattr("resolver.app.make_context", lambda c, offline, previews: (
        __import__("resolver.session", fromlist=["Context"]).Context(cfg=cfg, index=index, embedder=emb), []))
    code = cli.main(["eval", "--prompts", str(prompts), "--conditions", "A,B", "--out", str(tmp_path / "run"),
                     "--offline", "--hand-briefs", str(tmp_path / "none.jsonl")])
    assert code == 0 and (tmp_path / "run" / "results_B.json").exists()
    code = cli.main(["eval", "--prompts", str(prompts), "--score", str(tmp_path / "run"),
                     "--labels", str(tmp_path / "none.jsonl")])
    assert code == 0 and (tmp_path / "run" / "report.md").exists()
