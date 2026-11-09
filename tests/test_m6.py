"""M6: brief schema, rights guard, cache, offline and degraded paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from resolver import brief as brief_mod
from resolver.config import Config
from resolver.schemas import Brief, CueSpec, NamedReference

RECORDED = Path(__file__).parent / "recorded"
IN01 = ("take this video put a tarak mehta ka ulta chasma famous fighting or long sequence music in the "
        "background and put caption when claude limit is reached mid session")


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.library.cache_dir = str(tmp_path / "cache")
    return c


def recorded(*names):
    outputs = [(RECORDED / n).read_text() if n.endswith(".txt") else n for n in names]
    calls = []

    def call(system, user):
        calls.append((system, user))
        return outputs[min(len(calls), len(outputs)) - 1], ["https://example.org/search-hit"]

    call.calls = calls
    return call


def test_prompt_file_and_keyword_map_ship_with_package():
    assert "Output only one JSON object" in brief_mod.system_prompt("v1")
    data = brief_mod.keyword_map()
    assert "tarak mehta" in data["references"] and "fight" in data["keywords"]


def test_l8_rights_guard_strips_names_and_melody_words():
    cue = CueSpec(label="Tarak Mehta fight", mood=["comic", "same tune as the theme"], instruments=["dhol"],
                  queries=["Tarak Mehta style comic fight music", "dhol and tabla slapstick, Hans Zimmer brass",
                           "the exact melody of the theme", "lyrics about a fight", "fast comic percussion"])
    b = Brief(version="v1", mode="online", primary=cue,
              named_reference=NamedReference(raw="tarak mehta", canonical="Taarak Mehta Ka Ooltah Chashmah"))
    out = brief_mod.rights_guard(b, extra_names=["Hans Zimmer"])
    qs = [q.lower() for q in out.primary.queries]
    assert not any(t in q for q in qs for t in ("tarak", "mehta", "zimmer", "hans", "melody", "lyrics"))
    assert "style comic fight music" in qs and "fast comic percussion" in qs
    assert out.primary.mood == ["comic"] and "tarak" not in out.primary.label.lower()


def test_rights_guard_keeps_two_queries_via_fallback():
    cue = CueSpec(label="x", mood=["comic", "playful"], instruments=["tabla"], queries=["the melody of X", "same tune"])
    out = brief_mod.rights_guard(Brief(version="v1", mode="online", primary=cue),
                                 fallback_queries=brief_mod._fallback_queries)
    assert len(out.primary.queries) >= 2 and not any(brief_mod.has_banned(q) for q in out.primary.queries)


def test_r3_disjunction_gives_alternatives(cfg):
    b = brief_mod.make_brief(IN01, cfg, call=recorded("brief_in01.txt"))
    assert b.mode == "online" and len(b.alternatives) == 1
    assert b.named_reference.available is False and b.named_reference.canonical.startswith("Taarak")
    assert "https://example.org/search-hit" in b.named_reference.sources
    # The alternative's melody query was dropped by the guard; names never reach a query.
    for cue in b.cues():
        assert len(cue.queries) >= 2
        assert not any("mehta" in q.lower() or "tune" in q.lower() for q in cue.queries)


def test_r8_ignores_other_instructions(cfg):
    b = brief_mod.make_brief("add captions, cut the first two seconds, happy ukulele vlog music, colour grade warmer",
                             cfg, call=recorded("brief_r8.txt"))
    assert b.ignored_instructions == ["add captions", "cut the first two seconds", "colour grade warmer"]
    assert b.primary.instruments[0] == "ukulele"


def test_r8_offline_splits_off_captions(cfg):
    b = brief_mod.offline_brief(IN01, cfg)
    assert any("caption" in i for i in b.ignored_instructions)
    assert not any("caption" in q for c in b.cues() for q in c.queries)


def test_r3_offline_disjunction(cfg):
    b = brief_mod.offline_brief(IN01, cfg)
    assert len(b.alternatives) == 1 and b.primary.label != b.alternatives[0].label
    b2 = brief_mod.offline_brief("either chase music or something suspenseful, whichever fits better", cfg)
    assert b2.alternatives and "suspense" in b2.alternatives[0].mood


def test_r1_r2_offline_reference_normalised(cfg):
    b = brief_mod.offline_brief("tarak mehta ka ulta chasma fight music", cfg)
    assert b.named_reference.canonical == "Taarak Mehta Ka Ooltah Chashmah" and b.named_reference.available is False
    assert "dhol" in b.primary.instruments
    assert any("not available" in n for n in b.notes_for_user)
    assert not any("mehta" in q or "chasma" in q for q in b.primary.queries)


def test_r10_implied_and_r15_deferred(cfg):
    assert brief_mod.offline_brief("make it funny", cfg).music_requested == "implied"
    b = brief_mod.offline_brief("dramatic music first then switch to comic when he falls", cfg)
    assert b.deferred_cues and "comic" in b.deferred_cues[0] and "dramatic" in b.primary.mood


def test_l8_exact_copy_request_gets_genre_note(cfg):
    b = brief_mod.offline_brief("make it sound exactly like the pirates of the caribbean theme", cfg)
    assert any("genre-level" in n for n in b.notes_for_user)
    assert not any("exactly" in q or "pirates" in q or "caribbean" in q for q in b.primary.queries)


def test_d1_offline_brief_without_key(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    b = brief_mod.make_brief("sad violin like a 90s bollywood breakup scene", cfg)
    assert b.mode == "offline" and any("D1" in n for n in b.notes_for_user)
    assert "sad" in b.primary.mood and len(b.primary.queries) >= 2


def test_d5_retry_then_offline_on_bad_json(cfg):
    good_after_bad = recorded("not json at all", "brief_r8.txt")
    b = brief_mod.make_brief("happy ukulele vlog music", cfg, call=good_after_bad)
    assert b.mode == "online" and len(good_after_bad.calls) == 2
    assert "previous output was invalid" in good_after_bad.calls[1][1]

    bad = recorded("{\"primary\": {\"label\": 3}}", "still not json")
    b = brief_mod.make_brief("happy ukulele vlog music, take two", cfg, call=bad)
    assert len(bad.calls) == 2 and b.mode == "offline" and any("D5" in n for n in b.notes_for_user)


def test_d2_network_error_goes_offline(cfg):
    def down(system, user):
        raise ConnectionError("no route")
    b = brief_mod.make_brief("calm piano", cfg, call=down)
    assert b.mode == "offline" and any("D2" in n for n in b.notes_for_user)


def test_d8_brief_cached(cfg):
    call = recorded("brief_r8.txt")
    a = brief_mod.make_brief("Happy ukulele  vlog music", cfg, call=call)
    b = brief_mod.make_brief("happy ukulele vlog music", cfg, call=call)
    assert len(call.calls) == 1 and a == b
    cfg.brief.prompt_version = "v2"
    assert brief_mod.cache_key("x", "m", "v1") != brief_mod.cache_key("x", "m", "v2")


def test_r11_refine_is_a_delta(cfg):
    b = brief_mod.offline_brief("comic fight music with brass", cfg)
    r = brief_mod.refine(b, "more comic, no brass, faster", cfg)
    assert r.primary.mood[0] in {"playful", "quirky", "comic"}
    assert "brass" in r.primary.avoid and not any("brass" in i for i in r.primary.instruments)
    assert r.primary.tempo_bpm[0] >= (b.primary.tempo_bpm or (100, 130))[0] + 20
    assert r.named_reference == b.named_reference and r.ignored_instructions == b.ignored_instructions


def test_refine_directives():
    assert brief_mod.directives("like #2")["like_rank"] == 2
    assert brief_mod.directives("something completely different")["different"]
    assert brief_mod.directives("more comic") == {"like_rank": None, "different": False, "include_shown": False}


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_live_brief_tarak_mehta(cfg):
    b = brief_mod.make_brief(IN01, cfg, use_cache=False)
    assert b.mode == "online" and b.named_reference and b.named_reference.sources
    assert b.alternatives and b.ignored_instructions
    assert not any("mehta" in q.lower() for c in b.cues() for q in c.queries)
