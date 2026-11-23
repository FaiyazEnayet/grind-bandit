"""Unit tests for the shadow retention model (GB14).

Known-answer fixtures: exact recall probabilities, hand-computed
scores, and controlled simulation with a known ground-truth half-life
used only to validate the fitting machinery. Simulated rows are never
presented as real practice data.
"""

import math

import pytest

from grindbandit.research import retention as rt

T0 = 1_700_000_000.0
DAY = 86400.0


def _obs(topic="array", delta=1.0, solved=True, prior_successes=0, i=0):
    ts = T0 + i * DAY
    return {"topic": topic, "delta_days": delta, "solved": solved,
            "prior_successes": prior_successes, "ts": ts,
            "last_ts": ts - delta * DAY}


def test_recall_probability_known_values():
    assert rt.recall_probability(0.0, 30.0) == 1.0
    assert rt.recall_probability(30.0, 30.0) == pytest.approx(0.5)
    assert rt.recall_probability(60.0, 30.0) == pytest.approx(0.25)
    assert rt.recall_probability(15.0, 30.0) == pytest.approx(math.sqrt(0.5))
    with pytest.raises(ValueError):
        rt.recall_probability(1.0, 0.0)


def test_brier_and_log_loss_known_values():
    # Coin-flip predictions on one success and one failure.
    assert rt.brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)
    assert rt.log_loss([0.5, 0.5], [True, False]) == pytest.approx(math.log(2))
    # Perfect predictions score zero.
    assert rt.brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert rt.log_loss([1.0, 0.0], [True, False]) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError):
        rt.brier_score([], [])


def test_calibration_bins_known_values():
    preds = [0.05, 0.15, 0.85, 0.95]
    outcomes = [False, True, True, True]
    bins = rt.calibration_bins(preds, outcomes)
    by_low = {b["bin_low"]: b for b in bins}
    assert by_low[0.0]["n"] == 1
    assert by_low[0.0]["observed_rate"] == pytest.approx(0.0)
    assert by_low[0.1]["n"] == 1
    assert by_low[0.1]["observed_rate"] == pytest.approx(1.0)
    assert by_low[0.8]["n"] == 1
    assert by_low[0.9]["n"] == 1
    assert by_low[0.9]["mean_predicted"] == pytest.approx(0.95)


def test_due_date_error_known_values():
    # Fixed 30-day model, next attempt exactly 30 days after practice.
    model = {"kind": "fixed30", "half_life": 30.0}
    obs = [_obs(delta=30.0, i=0)]
    assert rt.due_date_error(model, obs) == pytest.approx(0.0)
    # Ten days late against a 30-day due date.
    obs = [_obs(delta=40.0, i=0)]
    assert rt.due_date_error(model, obs) == pytest.approx(10.0)
    # The global rate has no half-life and no due date.
    assert rt.due_date_error({"kind": "global_rate", "rate": 0.7}, obs) is None


def test_split_holdout_is_latest_quarter_by_time():
    obs = [_obs(i=i) for i in range(8)]
    train, holdout = rt.split_holdout(obs)
    assert len(train) == 6
    assert len(holdout) == 2
    assert [o["ts"] for o in holdout] == [T0 + 6 * DAY, T0 + 7 * DAY]
    train2, holdout2 = rt.split_holdout([])
    assert train2 == [] and holdout2 == []


def test_fit_global_half_life_recovers_known_truth():
    obs = rt.simulate_observations(2000, half_lives=12.0, seed=3)
    fit = rt.fit_global_half_life(obs)
    assert fit["half_life"] == pytest.approx(12.0, abs=1.5)
    assert fit["at_bound"] is False
    with pytest.raises(ValueError):
        rt.fit_global_half_life([])


def test_fit_global_half_life_all_failures_hits_lower_bound():
    obs = [_obs(delta=1.0 + 0.05 * i, solved=False, i=i) for i in range(50)]
    fit = rt.fit_global_half_life(obs)
    assert fit["at_bound"] is True
    assert fit["half_life"] == pytest.approx(rt.MIN_HALF_LIFE_DAYS, rel=1e-3)


def test_topic_shrinkage_pulls_sparse_topics_to_global():
    # A topic with 2 observations keeps the global half-life; a topic
    # with many observations lands between the global value and its own
    # maximum-likelihood fit in log space.
    obs = rt.simulate_observations(300, half_lives={"array": 6.0, "dp": 60.0}, seed=5)
    glob = rt.fit_global_half_life(obs)
    fit = rt.fit_topic_half_lives(obs, glob["half_life"])
    assert fit["half_lives"]["array"] < glob["half_life"]
    assert fit["half_lives"]["dp"] > glob["half_life"]
    sparse = [_obs(topic="rare", delta=2.0, solved=True, i=0),
              _obs(topic="rare", delta=2.0, solved=False, i=1)]
    fit2 = rt.fit_topic_half_lives(sparse, 20.0)
    assert fit2["half_lives"]["rare"] == pytest.approx(20.0)


def test_recurrence_model_is_monotone_in_prior_successes():
    obs = rt.simulate_observations(300, half_lives=20.0, seed=9)
    model = rt.fit_model("recurrence_h", obs)
    assert 0.0 <= model["b"] <= 1.0
    preds_low = rt.predict(model, [_obs(prior_successes=0, delta=10.0)])
    preds_high = rt.predict(model, [_obs(prior_successes=6, delta=10.0)])
    assert preds_high[0] >= preds_low[0]


def test_bootstrap_interval_is_zero_for_identical_predictions():
    preds = [0.6, 0.4, 0.7, 0.3]
    outcomes = [True, False, True, False]
    interval = rt.bootstrap_brier_difference(preds, preds, outcomes, resamples=50)
    assert interval["mean_diff"] == pytest.approx(0.0)
    assert interval["ci_low"] == pytest.approx(0.0)
    assert interval["ci_high"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        rt.bootstrap_brier_difference([], [], [])


def test_simulation_is_reproducible_and_labeled():
    first = rt.simulate_observations(50, half_lives=12.0, seed=11)
    second = rt.simulate_observations(50, half_lives=12.0, seed=11)
    assert first == second
    assert all(r.get("simulated") is True for r in first)
    third = rt.simulate_observations(50, half_lives=12.0, seed=12)
    assert third != first


def test_compare_models_not_identifiable_below_minimums():
    obs = rt.simulate_observations(40, half_lives=12.0, seed=7)
    train, holdout = rt.split_holdout(obs)
    report = rt.compare_models(train, holdout)
    assert report["verdict"] == "NOT_IDENTIFIABLE"
    assert report["n_train"] == 30
    assert report["n_holdout"] == 10


def test_predict_global_rate_is_constant():
    model = {"kind": "global_rate", "rate": 0.7}
    assert rt.predict(model, [_obs(), _obs(i=1)]) == [0.7, 0.7]
    with pytest.raises(ValueError):
        rt.fit_model("not_a_model", [_obs()])
