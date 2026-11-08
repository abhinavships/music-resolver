import pytest

from resolver.fallback import ChargeNotApproved, Fallback, Option


def test_skipping_everything_never_dead_ends():
    f = Fallback(generation_cost=1.99)
    assert f.current is Option.LICENSED
    assert f.skip() is Option.OWN_FILE
    assert f.skip("no file") is Option.GENERATE
    assert f.skip("declined") is Option.NONE
    assert f.finished and f.chosen is Option.NONE
    assert f.charged == 0.0
    assert "licensed: skipped by user" in f.summary()
    assert "generate: declined" in f.summary()


def test_no_match_moves_on_and_names_the_search():
    f = Fallback(generation_cost=1.99)
    assert f.no_match("fma-cc subset") is Option.OWN_FILE
    assert "fma-cc subset" in f.tried[0]


def test_generation_needs_approval_of_exact_cost():
    f = Fallback(generation_cost=1.99)
    f.skip()
    f.skip()
    with pytest.raises(ChargeNotApproved):
        f.pick()
    with pytest.raises(ChargeNotApproved):
        f.pick(approved_cost=0.99)
    assert f.charged == 0.0
    assert f.pick(approved_cost=1.99) is Option.GENERATE
    assert f.charged == 1.99


def test_free_options_charge_nothing():
    f = Fallback(generation_cost=1.99)
    assert f.pick() is Option.LICENSED
    assert f.charged == 0.0


def test_missing_provider_skips_generation_and_says_so():
    f = Fallback(generation_cost=None)
    f.skip()
    assert f.skip() is Option.NONE
    assert "no provider configured" in f.summary()


def test_finished_flow_rejects_more_actions():
    f = Fallback()
    f.pick()
    with pytest.raises(ValueError):
        f.skip()
