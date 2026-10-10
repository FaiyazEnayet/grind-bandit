"""Tests for the Thompson sampling linear bandit."""

import numpy as np

from grindbandit import bandit, mastery


def _problem(topics, difficulty="Medium"):
    return {"id": 1, "title": "T", "difficulty": difficulty, "topics": topics}


def test_context_features_for_a_new_user():
    s = {}
    x = bandit.context(_problem(["dp"]), s, 1_000_000.0)
    assert x.shape == (bandit.D,)
    assert x[0] == 0.5      # gap_min: 1 - 0.5 prior
    assert x[2] == 1.0      # difficulty_match: Medium fits 0.5 mastery
    assert x[3] == 1.0      # recency: never touched


def test_target_difficulty_thresholds():
    assert bandit.target_difficulty(0.2) == 0.0
    assert bandit.target_difficulty(0.5) == 0.5
    assert bandit.target_difficulty(0.9) == 1.0


def test_difficulty_match_shifts_with_mastery():
    s = {}
    for i in range(30):
        mastery.update(s, ["array"], True, float(i))
    easy = bandit.context(_problem(["array"], "Easy"), s, 1000.0)
    hard = bandit.context(_problem(["array"], "Hard"), s, 1000.0)
    # high mastery implies Hard is the right level now
    assert hard[2] > easy[2]


def test_observe_moves_posterior_mean():
    s = {}
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    bandit.observe(s, x, 0.5)
    mean, _ = bandit.get_posterior(s)
    assert np.all(mean > 0)


def test_sampling_is_deterministic_with_seed():
    s = {}
    t1 = bandit.sample_theta(s, seed=0)
    t2 = bandit.sample_theta(s, seed=0)
    np.testing.assert_array_equal(t1, t2)
    t3 = bandit.sample_theta(s, seed=1)
    assert not np.allclose(t1, t3)


def test_selection_prefers_weak_topics():
    s = {}
    # user is strong at arrays, weak at dp
    for i in range(10):
        mastery.update(s, ["array"], True, float(i))
    for i in range(10):
        mastery.update(s, ["dp"], False, float(i))
    weak = _problem(["dp"], "Medium")
    strong = _problem(["array"], "Medium")
    now = 10_000.0
    x_weak = bandit.context(weak, s, now)
    x_strong = bandit.context(strong, s, now)
    # a theta that values mastery gaps must score the weak problem higher
    theta = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    assert bandit.score(theta, x_weak) > bandit.score(theta, x_strong)


def test_posterior_learns_gap_weight_from_rewards():
    s = {}
    x = np.array([0.9, 0.9, 0.5, 1.0, 1.0])
    for _ in range(20):
        bandit.observe(s, x, 0.3)
    mean, _ = bandit.get_posterior(s)
    assert mean[0] > 0 and mean[1] > 0


def test_explain_mentions_weakest_topic():
    s = {}
    mastery.update(s, ["dp"], False, 0.0)
    lines = bandit.explain(_problem(["dp", "array"]), s, 1000.0)
    assert any("dp" in line for line in lines)
