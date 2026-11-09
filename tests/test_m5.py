"""M5: licence parsing and filtering, adapters, ingest idempotency, coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from resolver.config import Config
from resolver.ingest import ingest, load_index, segment_starts
from resolver.library import FilterStats, Stub, attribution, filter_stubs, parse_licence
from resolver.library.base import LicenceError
from resolver.library.jamendo_api import JamendoApi
from resolver.library.local_folder import LocalFolder
from resolver.library.mtg_jamendo import MtgJamendo
from resolver.schemas import TrackRecord

from toy import ToyEmbedder, build_toy_library, make_toy_track

RECORDED = Path(__file__).parent / "recorded"


@pytest.mark.parametrize("raw,code", [
    ("http://creativecommons.org/licenses/by/3.0/", "cc-by"),
    ("https://creativecommons.org/licenses/by-sa/4.0/", "cc-by-sa"),
    ("http://creativecommons.org/licenses/by-nc-sa/3.0/", "cc-by-nc-sa"),
    ("http://creativecommons.org/licenses/by-nc/2.5/", "cc-by-nc"),
    ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ("CC0", "cc0"), ("cc-by", "cc-by"), ("CC BY 4.0", "cc-by"), ("CC BY-NC-SA 4.0", "cc-by-nc-sa"),
    ("public domain", "pd"),
])
def test_parse_licence(raw, code):
    assert parse_licence(raw) == code


@pytest.mark.parametrize("raw", [
    "http://creativecommons.org/licenses/by-nd/3.0/", "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "CC BY-ND 4.0", "cc-by-nc-nd",
])
def test_l5_nd_always_excluded(raw):
    with pytest.raises(LicenceError) as err:
        parse_licence(raw)
    assert err.value.code == "nd"
    stats = FilterStats()
    for profile in ("demo", "commercial"):
        assert list(filter_stubs([Stub("x:1", "t", "a", "p", raw)], profile, stats)) == []
    assert stats.excluded_nd == 2


@pytest.mark.parametrize("raw", [None, "", "all rights reserved", "GPL", "https://example.org/licence", "cc-xyz"])
def test_l6_unknown_licence_excluded(raw):
    stats = FilterStats()
    assert list(filter_stubs([Stub("x:1", "t", "a", "p", raw)], "demo", stats)) == []
    assert stats.excluded_unknown == 1


def test_l3_l4_demo_only():
    stubs = [Stub(f"x:{c}", "t", "a", "p", c) for c in ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "pd")]
    demo = {code for _, code in filter_stubs(stubs, "demo", FilterStats())}
    stats = FilterStats()
    commercial = {code for _, code in filter_stubs(stubs, "commercial", stats)}
    assert demo == {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "pd"}
    assert commercial == {"cc0", "cc-by", "pd"}
    assert stats.excluded_profile == {"cc-by-sa": 1, "cc-by-nc": 1, "cc-by-nc-sa": 1}


def test_l2_attribution_line():
    t = TrackRecord(track_id="x:1", title="Dhol Brawl", artist="Artist A", duration=30, path="p",
                    source_url="https://example.org/1", licence="cc-by")
    line = attribution(t)
    assert '"Dhol Brawl" by Artist A' in line and "CC BY" in line
    assert "creativecommons.org/licenses/by/4.0" in line and "https://example.org/1" in line
    assert attribution(t.model_copy(update={"licence": "cc0"})) is None


def test_local_folder_adapter(tmp_path):
    lib = build_toy_library(tmp_path / "lib", duration=2.0)
    make_toy_track(lib / "unlisted.wav", ["comic"], duration=2.0)
    stubs = {s.track_id: s for s in LocalFolder(lib).stubs()}
    assert len(stubs) == 15
    s = stubs["local_folder:dhol_fight.wav"]
    assert (s.title, s.artist, s.licence_raw) == ("Dhol Brawl", "Artist A", "cc-by")
    assert "dhol" in s.tags
    assert stubs["local_folder:unlisted.wav"].licence_raw is None   # no row: L6


def test_jamendo_api_adapter_recorded(tmp_path):
    pages = json.loads((RECORDED / "jamendo_tracks.json").read_text())
    urls: list[str] = []

    def fetch(url):
        urls.append(url)
        return pages

    def download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        make_toy_track(dest.with_suffix(".wav"), ["dhol"], duration=1.0)
        dest.with_suffix(".wav").rename(dest)

    api = JamendoApi(tmp_path, tags=["indian"], limit=50, client_id="test", fetch=fetch, download=download)
    stubs = list(api.stubs())
    assert "client_id=test" in urls[0] and "tags=indian" in urls[0] and "musicinfo" in urls[0]
    ids = [s.track_id for s in stubs]
    assert "jamendo_api:1003" not in ids   # audiodownload_allowed = false
    by_id = {s.track_id: s for s in stubs}
    assert by_id["jamendo_api:1001"].licence_raw.endswith("/by/3.0/")
    assert "dhol" in by_id["jamendo_api:1001"].tags
    stats = FilterStats()
    kept = {s.track_id: code for s, code in filter_stubs(stubs, "commercial", stats)}
    assert kept == {"jamendo_api:1001": "cc-by"}
    assert stats.excluded_nd == 1 and stats.excluded_profile == {"cc-by-nc-sa": 1}


def test_jamendo_without_client_id_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)
    with pytest.raises(RuntimeError, match="JAMENDO_CLIENT_ID"):
        list(JamendoApi(tmp_path, tags=["x"]).stubs())


def test_mtg_jamendo_adapter(tmp_path):
    (tmp_path / "audio" / "14").mkdir(parents=True)
    (tmp_path / "autotagging.tsv").write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "track_0000214\tartist_000014\talbum_000031\t14/214.mp3\t124.6\tgenre---indian\tinstrument---tabla\n"
        "track_0000215\tartist_000014\talbum_000031\t14/215.mp3\t99.0\tgenre---rock\n"
        "track_0000216\tartist_000015\talbum_000032\t14/216.mp3\t80.0\tgenre---indian\n"
    )
    (tmp_path / "licences.tsv").write_text("track_0000214\thttp://creativecommons.org/licenses/by-nc/3.0/\n")
    stubs = list(MtgJamendo(tmp_path, tags=["indian"]).stubs())
    assert [s.track_id for s in stubs] == ["mtg_jamendo:track_0000214", "mtg_jamendo:track_0000216"]
    assert stubs[0].tags == ["indian", "tabla"] and stubs[0].licence_raw.endswith("by-nc/3.0/")
    assert stubs[1].licence_raw is None


def test_segment_starts():
    assert segment_starts(4.0, 10, 5, 360) == [0.0]
    assert segment_starts(30.0, 10, 5, 360) == [0, 5, 10, 15, 20]
    assert len(segment_starts(1000.0, 10, 5, 360)) == 71


@pytest.fixture
def toy_cfg(tmp_path):
    lib = build_toy_library(tmp_path / "lib", duration=12.0)
    cfg = Config()
    cfg.library.local_folder = str(lib)
    cfg.index.dir = str(tmp_path / "index")
    return cfg


def test_ingest_builds_index_and_coverage(toy_cfg):
    emb = ToyEmbedder()
    index = ingest(toy_cfg, [LocalFolder(toy_cfg.library.local_folder)], emb, log=lambda *_: None)
    assert index.size == 12                              # ND and unlicensed dropped
    assert "local_folder:nd_track.wav" not in index.tracks
    assert index.vectors.shape == (len(index.seg_tracks), emb.text(["x"]).shape[1])
    assert len(index.segments_of("local_folder:dhol_fight.wav")) == 1   # 12 s track: one 10 s window
    t = index.tracks["local_folder:dhol_fight.wav"]
    assert t.licence == "cc-by" and t.lufs is not None and t.bpm is not None
    assert index.tracks["local_folder:vocal_pop.wav"].vocal_prob > 0.9
    assert index.tracks["local_folder:ambient_calm.wav"].vocal_prob < 0.1

    cov = json.loads((Path(toy_cfg.index.dir) / "coverage.json").read_text())
    assert cov["tracks"] == 12
    assert cov["excluded"]["excluded_nd"] == 1 and cov["excluded"]["excluded_unknown"] == 1
    assert cov["per_licence"]["cc-by"]["tracks"] == 7
    assert cov["instruments"]["dhol"]["tracks"] == 2 and cov["instruments"]["dhol"]["method"] == "zero-shot"
    assert cov["instruments"]["shehnai"]["segments"] == 0


def test_ingest_idempotent(toy_cfg):
    adapters = [LocalFolder(toy_cfg.library.local_folder)]
    emb = ToyEmbedder()
    first = ingest(toy_cfg, adapters, emb, log=lambda *_: None)
    audio_calls = emb.calls["audio"]
    second = ingest(toy_cfg, adapters, emb, log=lambda *_: None)
    assert emb.calls["audio"] == audio_calls                     # nothing re-embedded
    assert second.seg_tracks == first.seg_tracks
    np.testing.assert_allclose(second.vectors, first.vectors, atol=1e-3)
    reloaded = load_index(toy_cfg.index.dir)
    assert reloaded.size == first.size and reloaded.model_id == "toy-tones-v1"


def test_ingest_profile_change_drops_tracks(toy_cfg):
    adapters = [LocalFolder(toy_cfg.library.local_folder)]
    ingest(toy_cfg, adapters, ToyEmbedder(), log=lambda *_: None)
    toy_cfg.library.profile = "commercial"
    index = ingest(toy_cfg, adapters, ToyEmbedder(), log=lambda *_: None)
    assert {t.licence for t in index.tracks.values()} <= {"cc0", "cc-by", "pd"}
    assert set(index.seg_tracks) == set(index.tracks)


def test_t7_corrupt_file_skipped(toy_cfg):
    bad = Path(toy_cfg.library.local_folder) / "broken.mp3"
    bad.write_bytes(b"not audio at all")
    with open(Path(toy_cfg.library.local_folder) / "licences.csv", "a") as fh:
        fh.write("broken.mp3,Broken,Artist Z,cc0,,,comic\n")
    index = ingest(toy_cfg, [LocalFolder(toy_cfg.library.local_folder)], ToyEmbedder(), log=lambda *_: None)
    assert "local_folder:broken.mp3" not in index.tracks
    assert index.coverage["excluded"]["undecodable"] == 1


def test_d3_tag_mode_without_embedder(toy_cfg):
    index = ingest(toy_cfg, [LocalFolder(toy_cfg.library.local_folder)], None, log=lambda *_: None)
    assert index.tag_mode and index.model_id == "tags"
    assert len(index.seg_tracks) == index.size == 12
    assert index.coverage["instruments"]["dhol"] == {"tracks": 2, "segments": 2, "method": "tags"}
    assert index.tracks["local_folder:vocal_pop.wav"].vocal_prob == 1.0
