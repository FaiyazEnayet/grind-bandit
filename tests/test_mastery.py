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
