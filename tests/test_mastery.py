"""Tests for the Beta Bernoulli mastery model."""

from grindbandit import mastery


def test_new_topic_starts_at_fifty_percent():
    assert mastery.mean({}, "dp") == 0.5


def test_solved_raises_mastery():
    s = {}
    mastery.update(s, ["dp"], True, 0.0)
    assert mastery.mean(s, "dp") > 0.5


def test_failed_lowers_mastery():
    s = {}
    mastery.update(s, ["dp"], False, 0.0)
    assert mastery.mean(s, "dp") < 0.5


def test_update_touches_every_topic_on_the_problem():
    s = {}
    mastery.update(s, ["dp", "array"], True, 0.0)
    assert mastery.mean(s, "dp") > 0.5
    assert mastery.mean(s, "array") > 0.5


def test_mastery_converges_with_evidence():
    s = {}
    for i in range(20):
        mastery.update(s, ["dp"], True, float(i))
    assert mastery.mean(s, "dp") > 0.9


def test_replay_rebuilds_history():
    log = [
        {"ts": 100.0, "topics": ["dp"], "solved": True},
        {"ts": 200.0, "topics": ["dp"], "solved": False},
        {"ts": 300.0, "topics": ["dp"], "solved": True},
    ]
    early = mastery.replay(log, 150.0)
    late = mastery.replay(log, 400.0)
    assert early["dp"] > 0.5
    assert late["dp"] != early["dp"]
    assert mastery.replay(log, 50.0) == {}


DAY = 86400.0


def test_no_decay_when_no_time_passes():
    s = {}
    mastery.update(s, ["dp"], True, 0.0)
    assert mastery.mean(s, "dp", 0.0) == mastery.mean(s, "dp")
    assert mastery.mean(s, "dp", 0.0) == 2.0 / 3.0


def test_decay_exact_half_life_math():
    # one solve: alpha=2, beta=1. After exactly one half-life (30d),
    # excess counts halve: alpha=1.5, beta=1 -> mean 0.6.
    s = {}
    mastery.update(s, ["dp"], True, 0.0)
    assert mastery.mean(s, "dp", 30 * DAY) == 0.6


def test_decay_drifts_toward_fifty_percent():
    s = {}
    for i in range(20):
        mastery.update(s, ["dp"], True, float(i))
    assert mastery.mean(s, "dp", 0.0) > 0.9
    late = mastery.mean(s, "dp", 300 * DAY)
    assert 0.5 < late < 0.6


def test_decay_never_crosses_the_prior():
    s = {}
    for i in range(10):
        mastery.update(s, ["dp"], True, float(i))
    mid = mastery.mean(s, "dp", 300 * DAY)
    assert 0.5 < mid < 0.6
    assert mastery.mean(s, "dp", 10_000 * DAY) == 0.5  # full return to prior
    s2 = {}
    for i in range(10):
        mastery.update(s2, ["dp"], False, float(i))
    mid2 = mastery.mean(s2, "dp", 300 * DAY)
    assert 0.4 < mid2 < 0.5
    assert mastery.mean(s2, "dp", 10_000 * DAY) == 0.5


def test_untouched_topic_ignores_now():
    s = {}
    assert mastery.mean(s, "dp", 10_000 * DAY) == 0.5


def test_config_half_life_override():
    s = {"config": {"half_life_days": 10.0}}
    mastery.update(s, ["dp"], True, 0.0)
    assert mastery.mean(s, "dp", 10 * DAY) == 0.6
    # default 30d half-life would give 2/3 decayed less: mean > 0.6
    s2 = {}
    mastery.update(s2, ["dp"], True, 0.0)
    assert mastery.mean(s2, "dp", 10 * DAY) > 0.6


def test_replay_applies_decay_at_upto_ts():
    log = [
        {"ts": 0.0, "topics": ["dp"], "solved": True},
        {"ts": 0.0, "topics": ["dp"], "solved": True},
    ]
    fresh = mastery.replay(log, 0.0)
    stale = mastery.replay(log, 30 * DAY)
    assert fresh["dp"] == 3.0 / 4.0
    assert stale["dp"] == 2.0 / 3.0  # alpha 3 -> 1 + 2*0.5 = 2
    assert stale["dp"] < fresh["dp"]
