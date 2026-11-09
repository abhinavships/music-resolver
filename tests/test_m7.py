"""M7: retrieval on a toy index, session flow, exclusions, music-slot export, effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from resolver import retrieve as retrieve_mod
from resolver import session as session_mod
from resolver import sfx as sfx_mod
from resolver import clip as clip_stage
from resolver.config import Config
from resolver.ingest import ingest
from resolver.library import LocalFolder
from resolver.schemas import Brief, ClipAnalysis, CueSpec, HitPoint, Region

from toy import ToyEmbedder, build_toy_library


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("toylib")
    lib = build_toy_library(root / "lib", duration=30.0)
    cfg = Config()
    cfg.library.local_folder = str(lib)
    cfg.index.dir = str(root / "index")
    cfg.library.cache_dir = str(root / "cache")
    cfg.no_match.z_threshold = 1.5
    emb = ToyEmbedder()
    index = ingest(cfg, [LocalFolder(lib)], emb, log=lambda *_: None)
    tag_cfg = cfg.model_copy(deep=True)
    tag_cfg.index.dir = str(root / "tag_index")
    tag_index = ingest(tag_cfg, [LocalFolder(lib)], None, log=lambda *_: None)
    return cfg, index, emb, tag_index


def cue(label, queries, **kw):
    return CueSpec(label=label, queries=queries, **kw)


def brief_of(primary, *alts):
    return Brief(version="v1", mode="offline", primary=primary, alternatives=list(alts))


DHOL = cue("comic dhol fight", ["comic dhol fight percussion", "fast dhol and tabla comic", "comic fight tabla"],
           instruments=["dhol", "tabla"], mood=["comic"], energy="high")
NO_CLIP = None


def titles(cands):
    return [c.track.title for c in cands]


def test_b1_strong_matches_with_reasons(toy):
    cfg, index, emb, _ = toy
    status, cands, searched, _ = retrieve_mod.retrieve(brief_of(DHOL), index, emb, NO_CLIP, cfg)
    assert status == "ok" and len(cands) == 3
    assert cands[0].track.title == "Dhol Brawl"
    assert "dhol" in cands[0].reason and cands[0].attribution and cands[0].track.licence == "cc-by"
    # Attributes the embedder cannot hear (flat across the library) are never claimed.
    flat = cue("x", ["comic dhol"], instruments=["dhol"], mood=["slapstick"])
    assert retrieve_mod.reasons(flat, index.segments_of("local_folder:sitar_romance.wav")[0],
                                retrieve_mod.Scorer(index, emb)) == []
    assert searched["library_tracks"] == 12 and searched["queries"]["comic dhol fight"] == DHOL.queries
    assert "No Derivs" not in titles(cands) and "No Licence" not in titles(cands)


def test_b2_b3_no_match_status(toy, tmp_path):
    cfg, index, emb, _ = toy
    nothing = cue("carnatic chiptune", ["carnatic mridangam over chiptune", "8-bit chiptune konnakol"])
    status, cands, searched, _ = retrieve_mod.retrieve(brief_of(nothing), index, emb, NO_CLIP, cfg)
    assert status == "no_match" and cands == []
    assert searched["matched"]["carnatic chiptune"] is False and searched["z"]["carnatic chiptune"] < 1.5


def test_b5_one_track_per_artist(toy):
    cfg, index, emb, _ = toy
    # Artist A has both of the best tabla/comic tracks.
    q = cue("comic tabla", ["comic tabla harmonium", "playful tabla comic", "dhol tabla comic"])
    _, cands, _, _ = retrieve_mod.retrieve(brief_of(q), index, emb, NO_CLIP, cfg)
    artists = [c.track.artist for c in cands]
    assert len(artists) == len(set(artists))
    cfg2 = cfg.model_copy(deep=True)
    cfg2.retrieve.max_per_artist = 2
    _, cands2, _, _ = retrieve_mod.retrieve(brief_of(q), index, emb, NO_CLIP, cfg2)
    assert sum(c.track.artist == "Artist A" for c in cands2) == 2


def test_r3_slots_cover_alternatives(toy):
    cfg, index, emb, _ = toy
    calm = cue("calm ambient bed", ["calm ambient synth", "slow calm ambient"], mood=["calm"])
    suspense = cue("suspense", ["suspense orchestral dramatic", "dramatic suspense"], mood=["suspense"])
    _, cands, _, _ = retrieve_mod.retrieve(brief_of(DHOL, calm, suspense), index, emb, NO_CLIP, cfg)
    labels = [c.cue_label for c in cands]
    assert labels == ["comic dhol fight", "calm ambient bed", "suspense"]
    assert titles(cands)[1] == "Calm Drift" and titles(cands)[2] == "Door Creak"


def test_t2_vocal_penalty_under_dialogue(toy):
    cfg, index, emb, _ = toy
    q = cue("playful synth", ["playful synth", "playful synth pop"])
    talky = ClipAnalysis(path="x", duration=10, has_audio=True, speech_regions=[Region(start=0, end=8)])
    _, quiet, _, _ = retrieve_mod.retrieve(brief_of(q), index, emb, None, cfg)
    _, under, _, _ = retrieve_mod.retrieve(brief_of(q), index, emb, talky, cfg)
    s_quiet = next(c.score for c in quiet if c.track.title == "Sing Along")
    s_under = next((c.score for c in under if c.track.title == "Sing Along"), None)
    assert s_under is None or s_under < s_quiet - 0.1
    if s_under is not None:
        assert any("vocals" in w for c in under if c.track.title == "Sing Along" for w in c.warnings)


def test_t6_tempo_penalty():
    cfg = Config()
    assert retrieve_mod.tempo_penalty(120, (110, 130), cfg) == 0
    assert retrieve_mod.tempo_penalty(60, (110, 130), cfg) == 0         # half-time reading
    assert retrieve_mod.tempo_penalty(150, (110, 130), cfg) == pytest.approx(0.2)
    assert retrieve_mod.tempo_penalty(300, (100, 110), cfg) == cfg.retrieve.tempo_penalty_max


def test_l3_l4_warnings_reach_candidates(toy):
    cfg, index, emb, _ = toy
    q = cue("brass action", ["brass action epic", "epic fast brass"])
    _, cands, _, _ = retrieve_mod.retrieve(brief_of(q), index, emb, NO_CLIP, cfg)
    brass = next(c for c in cands if c.track.title == "Brass Charge")
    assert any("BY-SA" in w for w in brass.warnings)


def test_d3_tag_search_without_clap(toy):
    cfg, _, _, tag_index = toy
    status, cands, searched, _ = retrieve_mod.retrieve(brief_of(DHOL), tag_index, None, NO_CLIP, cfg)
    assert status == "ok" and cands[0].track.title in {"Dhol Brawl", "Dhol Party", "Tabla Tickle"}
    assert "BM25" in searched["mode"] and "dhol" in cands[0].reason


def test_b4_coverage_warning(toy):
    cfg, index, emb, _ = toy
    _, _, _, cov = retrieve_mod.retrieve(brief_of(DHOL), index, emb, NO_CLIP, cfg)
    assert any("dhol" in w and "B4" in w for w in cov)


# --- session ---------------------------------------------------------------------------------

def ctx_for(toy, **kw):
    cfg, index, emb, _ = toy
    return session_mod.Context(cfg=cfg, index=index, embedder=emb, offline_brief=True, previews=False, **kw)


@pytest.fixture(scope="module")
def clip_analysis(clip_path):
    return clip_stage.analyse(clip_path, Config())


def test_session_resolve_and_r11_exclusions(toy, clip_path, clip_analysis, tmp_path):
    ctx = ctx_for(toy)
    res = session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx, analysis=clip_analysis)
    assert res.status == "ok" and (tmp_path / "session.json").exists() and (tmp_path / "candidates.json").exists()
    first = {c.track.track_id for c in res.candidates}
    res2 = session_mod.refine(tmp_path, "more comic", ctx)
    assert res2.status == "ok" and res2.candidates
    assert not first & {c.track.track_id for c in res2.candidates}          # R11: shown tracks excluded
    state = json.loads((tmp_path / "session.json").read_text())
    assert [h["action"] for h in state["history"]] == ["resolve", "refine: more comic"]


def test_r13_completely_different(toy, clip_analysis, clip_path, tmp_path):
    ctx = ctx_for(toy)
    cfg = ctx.cfg.model_copy(deep=True)
    cfg.session.different_neighbours = 3
    ctx.cfg = cfg
    res = session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx, analysis=clip_analysis)
    shown = [c.track.track_id for c in res.candidates]
    near = retrieve_mod.neighbours(shown, retrieve_mod.Scorer(ctx.index, ctx.embedder), 3)
    res2 = session_mod.refine(tmp_path, "something completely different", ctx)
    got = {c.track.track_id for c in res2.candidates}
    assert not got & set(shown) and not got & near


def test_r12_more_like_this(toy, clip_analysis, clip_path, tmp_path):
    ctx = ctx_for(toy)
    res = session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx, analysis=clip_analysis)
    second = res.candidates[1]
    shown = {c.track.track_id for c in res.candidates}
    res2 = session_mod.refine(tmp_path, "like #2", ctx)
    assert res2.searched["audio_query"] and res2.candidates
    # The top result is the unshown track whose best segment sounds most like #2's segment.
    index = ctx.index
    target = session_mod._segment_vector(second, ctx)
    sims = index.vectors @ target
    best = {}
    for i, t in enumerate(index.seg_tracks):
        if t not in shown:
            best[t] = max(best.get(t, -1.0), float(sims[i]))
    assert res2.candidates[0].track.track_id == max(best, key=best.get)
    assert not shown & {c.track.track_id for c in res2.candidates}


def test_session_flow_resolve_skip_own(toy, clip_analysis, clip_path, track_path, tmp_path):
    ctx = ctx_for(toy)
    session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx, analysis=clip_analysis)
    res = session_mod.skip(tmp_path, ctx, "none fit")
    assert res.status == "needs_input" and res.fallback_option == "own_file"
    with pytest.raises(PermissionError):
        session_mod.own(tmp_path, str(track_path), ctx, rights_confirmed=False)
    report, paths = session_mod.own(tmp_path, str(track_path), ctx, rights_confirmed=True)
    assert Path(paths["video"]).exists() and report.status in {"ok", "verify_failed"}
    assert any("L7" in s for s in report.skipped)
    assert any("licensed: none fit" in s for s in report.skipped)          # fallback summary in the report
    assert next(c for c in report.checks if c.name == "licence_allowed").passed


def test_session_pick_renders_and_checks_licence(toy, clip_analysis, clip_path, tmp_path):
    ctx = ctx_for(toy)
    res = session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx, analysis=clip_analysis)
    report, paths = session_mod.pick(tmp_path, 1, ctx)
    names = {c.name: c for c in report.checks}
    assert names["licence_allowed"].passed and names["attribution_present"].passed
    assert report.attribution and res.candidates[0].track.title in report.attribution
    assert Path(paths["video"]).exists() and (tmp_path / "report.json").exists()
    plan = json.loads((tmp_path / "mix.json").read_text())
    assert plan["track_path"] == res.candidates[0].track.path


def test_l9_music_slot_export(toy, clip_analysis, clip_path, tmp_path):
    ctx = ctx_for(toy)
    session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx, analysis=clip_analysis)
    session_mod.skip(tmp_path, ctx)                 # licensed -> own file
    res = session_mod.skip(tmp_path, ctx)           # own file -> (no generator) -> none
    assert res.fallback_option == "none"
    marker = json.loads((tmp_path / "hit_marker.json").read_text())
    assert marker["hit_time"] == pytest.approx(clip_analysis.hits[0].time)
    assert (tmp_path / "output.mp4").exists()
    report = json.loads((tmp_path / "report.json").read_text())
    assert any("hit_marker.json" in s for s in report["skipped"])


def test_no_match_advances_fallback(toy, clip_analysis, clip_path, tmp_path):
    ctx = ctx_for(toy)
    res = session_mod.resolve(str(clip_path), "carnatic violin over a chiptune beat", tmp_path, ctx,
                              analysis=clip_analysis)
    assert res.status == "no_match" and res.fallback_option == "own_file"
    assert "own file" in res.message and res.searched["library_tracks"] == 12


class FakeProvider:
    name = "fake"

    def quote(self, brief, duration):
        return 0.5

    def generate(self, brief, duration, out):
        from toy import make_toy_track
        make_toy_track(out, ["comic"], duration=duration + 1)
        return out


def test_generate_needs_exact_approval(toy, clip_analysis, clip_path, tmp_path):
    ctx = ctx_for(toy, provider=FakeProvider())
    session_mod.resolve(str(clip_path), "comic dhol fight music", tmp_path, ctx, analysis=clip_analysis)
    session_mod.skip(tmp_path, ctx)
    res = session_mod.skip(tmp_path, ctx)
    assert res.fallback_option == "generate" and res.quoted_cost == 0.5
    refused = session_mod.generate(tmp_path, ctx)
    assert refused.status == "needs_input" and refused.quoted_cost == 0.5
    assert session_mod.generate(tmp_path, ctx, approved_cost=0.4).status == "needs_input"
    report, paths = session_mod.generate(tmp_path, ctx, approved_cost=0.5)
    assert Path(paths["video"]).exists()
    assert json.loads((tmp_path / "session.json").read_text())["fallback"]["charged"] == 0.5


# --- effects (F7) ---------------------------------------------------------------------------

def test_f7_sfx_on_secondary_hits(tmp_path):
    sr = 48000
    folder = tmp_path / "sfx"
    folder.mkdir()
    y = np.zeros(int(0.4 * sr), dtype=np.float32)
    y[int(0.1 * sr)] = 1.0                                   # the effect's peak is 0.1 s in
    y[int(0.1 * sr) + 1: int(0.3 * sr)] = 0.2 * np.exp(-np.linspace(0, 8, int(0.2 * sr) - 1))
    sf.write(folder / "whoosh.wav", y, sr)
    sf.write(folder / "paid.wav", y, sr)
    (folder / "licences.csv").write_text("file,title,artist,licence,licence_url,source_url,tags\n"
                                         "whoosh.wav,Whoosh,anon,cc0,,,whoosh\npaid.wav,Paid,x,cc-by,,,hit\n")
    assert [p.name for p in sfx_mod.cc0_effects(folder)] == ["whoosh.wav"]   # CC0 only

    clip = ClipAnalysis(path="x", duration=6, has_audio=True, hits=[
        HitPoint(time=3.0, score=0.9, audio_score=0.9, motion_score=0.9),
        HitPoint(time=1.5, score=0.5, audio_score=0.5, motion_score=0.5),
        HitPoint(time=4.5, score=0.4, audio_score=0.4, motion_score=0.4),
        HitPoint(time=5.5, score=0.05, audio_score=0.05, motion_score=0.05),
    ])
    cfg = Config()
    assert sfx_mod.plan_sfx(clip, cfg) == []                  # off by default
    cfg.sfx.enabled, cfg.sfx.folder = True, str(folder)
    plan = sfx_mod.plan_sfx(clip, cfg)
    assert [p["time"] for p in plan] == [1.5, 4.5]            # secondary hits only, weak one dropped
    stem = sfx_mod.render_sfx(plan, 6 * sr, sr, tmp_path / "work")
    mono = np.abs(stem).max(axis=1)
    for t in (1.5, 4.5):
        window = mono[int((t - 0.2) * sr): int((t + 0.2) * sr)]
        assert abs(np.argmax(window) / sr - 0.2) < 0.002     # peak on the hit
        assert 20 * np.log10(window.max()) == pytest.approx(cfg.sfx.gain_db, abs=0.1)


@pytest.mark.live
def test_live_clap_text_audio_alignment():
    """Real CLAP (downloads ~600 MB once): a tone and noise are told apart by text."""
    from resolver.embed import ClapEmbedder, clap_available
    if not clap_available():
        pytest.skip("torch/transformers not installed")
    e = ClapEmbedder()
    t = np.arange(e.sr * 10) / e.sr
    audio = e.audio([0.3 * np.sin(2 * np.pi * 440 * t), 0.1 * np.random.default_rng(0).standard_normal(t.size)])
    text = e.text(["a pure sine wave tone", "white noise static"])
    sims = audio @ text.T
    assert sims[0, 0] > sims[0, 1] and sims[1, 1] > sims[1, 0]
    assert float(text[0] @ text[1]) < 0.9          # not the degenerate larger_clap_music behaviour
