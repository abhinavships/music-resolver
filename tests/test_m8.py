"""M8: agent tool JSON contract, review page with ratings export, MCP wrapper."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from resolver import clip as clip_stage
from resolver import session as session_mod
from resolver import tool
from resolver.config import Config
from resolver.ingest import ingest
from resolver.library import LocalFolder
from resolver.review import write_blind_review, write_review
from resolver.schemas import ResolveResult

from toy import ToyEmbedder, build_toy_library


class Quote:
    name = "fake"

    def quote(self, brief, duration):
        return 1.25

    def generate(self, brief, duration, out):
        from toy import make_toy_track
        make_toy_track(out, ["comic"], duration=duration + 1)
        return out


@pytest.fixture(scope="module")
def env(tmp_path_factory, clip_path):
    root = tmp_path_factory.mktemp("m8")
    lib = build_toy_library(root / "lib", duration=30.0)
    cfg = Config()
    cfg.library.local_folder, cfg.index.dir, cfg.library.cache_dir = str(lib), str(root / "index"), str(root / "c")
    cfg.no_match.z_threshold = 1.5
    emb = ToyEmbedder()
    index = ingest(cfg, [LocalFolder(lib)], emb, log=lambda *_: None)
    analysis = clip_stage.analyse(clip_path, cfg)
    return cfg, index, emb, analysis


def ctx(env, previews=False, provider=None):
    cfg, index, emb, _ = env
    return session_mod.Context(cfg=cfg, index=index, embedder=emb, offline_brief=True, previews=previews,
                               provider=provider)


@pytest.fixture
def resolved(env, clip_path, tmp_path, monkeypatch):
    analysis = env[3]
    monkeypatch.setattr(clip_stage, "analyse", lambda *a, **k: analysis)
    c = ctx(env, provider=Quote())
    resp = tool.handle({"action": "resolve", "clip": str(clip_path), "prompt": "comic dhol fight music",
                        "session": str(tmp_path)}, c)
    return c, tmp_path, resp


def test_tool_resolve(resolved):
    _, out, resp = resolved
    assert resp["status"] == "ok" and len(resp["candidates"]) == 3
    ResolveResult.model_validate({k: v for k, v in resp.items() if k != "review_page"})
    assert Path(resp["review_page"]).exists() and resp["fallback_option"] == "licensed"


def test_tool_refine(resolved):
    c, out, first = resolved
    resp = tool.handle({"action": "refine", "session": str(out), "instruction": "more comic"}, c)
    assert resp["status"] == "ok"
    assert not {x["track"]["track_id"] for x in first["candidates"]} & {x["track"]["track_id"] for x in resp["candidates"]}


def test_tool_pick_with_override(resolved):
    c, out, _ = resolved
    resp = tool.handle({"action": "pick", "session": str(out), "rank": 1, "overrides": {"hit": 6.5}}, c)
    assert resp["status"] in {"ok", "verify_failed"} and Path(resp["outputs"]["video"]).exists()
    plan = json.loads(Path(resp["outputs"]["mix_plan"]).read_text())
    assert plan["hit_time"] == 6.5
    assert {"licence_allowed", "attribution_present"} <= {ch["name"] for ch in resp["report"]["checks"]}


def test_tool_skip_own_needs_rights(resolved, track_path):
    c, out, _ = resolved
    resp = tool.handle({"action": "skip", "session": str(out), "reason": "none fit"}, c)
    assert resp["status"] == "needs_input" and resp["fallback_option"] == "own_file"
    resp = tool.handle({"action": "own", "session": str(out), "track": str(track_path)}, c)
    assert resp["status"] == "needs_input" and "rights_confirmed" in resp["message"]
    resp = tool.handle({"action": "own", "session": str(out), "track": str(track_path), "rights_confirmed": True}, c)
    assert resp["status"] in {"ok", "verify_failed"} and resp["report"]["checks"]


def test_tool_generate_refuses_without_approved_cost(resolved):
    c, out, _ = resolved
    tool.handle({"action": "skip", "session": str(out)}, c)
    resp = tool.handle({"action": "skip", "session": str(out)}, c)
    assert resp["fallback_option"] == "generate" and resp["quoted_cost"] == 1.25
    for req in ({}, {"approved_cost": None}, {"approved_cost": 1.0}, {"approved_cost": "1.25"}):
        r = tool.handle({"action": "generate", "session": str(out), **req}, c)
        assert r["status"] == "needs_input" and r["quoted_cost"] == 1.25
    state = json.loads((out / "session.json").read_text())
    assert state["fallback"]["charged"] == 0 and state["fallback"]["chosen"] is None
    r = tool.handle({"action": "generate", "session": str(out), "approved_cost": 1.25}, c)
    assert r["status"] in {"ok", "verify_failed"}
    assert json.loads((out / "session.json").read_text())["fallback"]["charged"] == 1.25


def test_tool_errors_are_json(env, tmp_path):
    c = ctx(env)
    assert tool.handle({"action": "dance"}, c)["status"] == "error"
    assert "missing" in tool.handle({"action": "resolve", "session": str(tmp_path)}, c)["message"]
    assert "no session" in tool.handle({"action": "skip", "session": str(tmp_path / "nope")}, c)["message"]
    out = io.StringIO()
    assert tool.main(io.StringIO("{not json"), out) == 3 and json.loads(out.getvalue())["status"] == "error"


def test_mcp_server_exposes_tools():
    pytest.importorskip("mcp")
    import asyncio
    from resolver.mcp_server import build
    names = {t.name for t in asyncio.run(build().list_tools())}
    assert names == {"resolve_music", "refine_music", "pick_music", "skip_music", "own_music"}


def _chromium():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _chromium(), reason="Playwright Chromium not installed")
def test_review_page_previews_and_ratings(env, clip_path, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright
    monkeypatch.setattr(clip_stage, "analyse", lambda *a, **k: env[3])
    session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx(env, previews=True),
                        analysis=env[3])
    page_path = write_review(tmp_path)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(accept_downloads=True)
        page.goto(page_path.as_uri())
        videos = page.locator("video")
        assert videos.count() == 3
        for i in range(3):
            assert (tmp_path / videos.nth(i).get_attribute("src")).exists()
        page.wait_for_function("document.querySelector('video').readyState >= 1", timeout=10000)
        assert page.evaluate("document.querySelector('video').duration") > 5
        for slot, value in zip("ABC", (5, 3, 1)):
            page.locator(f'.card[data-slot="{slot}"] label').nth(value - 1).click()
        page.fill("#rater", "r1")
        with page.expect_download() as dl:
            page.click("#dl")
        data = json.loads(Path(dl.value.path()).read_text())
        browser.close()
    assert data["rater"] == "r1" and [r["rating"] for r in data["ratings"]] == [5, 3, 1]
    assert data["ratings"][0]["track_id"] and data["prompt"] == "comic dhol fight music"


def test_blind_review_hides_labels_and_shuffles(env, clip_path, tmp_path):
    session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx(env), analysis=env[3])
    res = ResolveResult.model_validate(json.loads((tmp_path / "session.json").read_text())["last_result"])
    p1 = write_blind_review([("A", res), ("B", res)], tmp_path, "prompt", rater_seed=1)
    p2 = write_blind_review([("A", res), ("B", res)], tmp_path, "prompt", rater_seed=2)
    html = p1.read_text()
    assert "Dhol" not in html and "Brief" not in html and "condition" not in html.lower()
    k1 = json.loads((tmp_path / "key-rater1.json").read_text())
    k2 = json.loads((tmp_path / "key-rater2.json").read_text())
    assert sorted(k1.values()) == ["A"] * 3 + ["B"] * 3 and k1 != k2
    assert p2.exists()
