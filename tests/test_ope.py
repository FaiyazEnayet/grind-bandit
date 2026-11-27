"""Stage 3 GB19: honest OPE feasibility/identifiability gate.

Logged estimated propensities alone do not validate inverse propensity
scoring. The diagnostics in ``grindbandit.research.ope_diagnostics``
run first and refuse to estimate whenever the log cannot support the
target policy: unlinked exposure, missing support, uncheckable
propensity Monte Carlo error, extreme importance weights, a collapsed
effective sample size, time leakage, or missing outcomes. Only when
every diagnostic passes does ``grindbandit.research.ope_estimators``
return IPS, self-normalized IPS, and doubly robust values, each with
a standard error and a prespecified Monte Carlo uncertainty bound.

The gate is validated on synthetic logged data with a known ground
truth, never on real practice data: the scripted demo history holds
30 attempts and no delayed quizzes, so nothing real is identifiable
here. Simulation validates the machinery only.
"""

import math
import random

from grindbandit.research import ope_diagnostics as od
from grindbandit.research import ope_estimators as oe

# True mean reward per action in the synthetic world, fixed before
# any estimate is looked at. Action C is the best action; the target
# policies below shift mass toward it.
TRUE_MEANS = {"A": 0.2, "B": 0.5, "C": 0.8}

# Prespecified Monte Carlo uncertainty bound: an estimate matches the
# known expected reward when it lands within this many estimator
# standard errors of the truth, plus a small allowance for the
# add-one smoothing shift in the logged propensities.
Z_BOUND = 3.0
SMOOTHING_ALLOWANCE = 0.005

NOW = 1_750_000_000.0


def _simulate(logging_probs, target_probs, n=20000, seed=7,
              samples=10000):
    """Logged rows from a known logging policy and target policy.

    Each row draws an action from ``logging_probs``, a Bernoulli
    reward with the action's true mean, and a logged propensity equal
    to the true logging probability (the add-one smoothed Monte Carlo
    estimate behind it is summarized by its sample count, exactly as
    GB3 records carry ``samples`` on the recommendation record).
    The target policy value is the exact sum over actions.
    """
    rng = random.Random(seed)
    actions = sorted(logging_probs)
    truth = sum(target_probs[a] * TRUE_MEANS[a] for a in actions)
    rows = []
    for i in range(n):
        draw = rng.random()
        cumulative = 0.0
        action = actions[-1]
        for a in actions:
            cumulative += logging_probs[a]
            if draw < cumulative:
                action = a
                break
        reward = 1.0 if rng.random() < TRUE_MEANS[action] else 0.0
        rows.append({
            "attempt_id": f"att-{i}",
            "recommendation_id": f"rec-{i}",
            "action": action,
            "logging_probability": logging_probs[action],
            "logging_probability_samples": samples,
            "target_probability": target_probs[action],
            "reward": reward,
            "exposure_ts": NOW + i,
            "outcome_ts": NOW + i + 3600.0,
        })
    recommendations = {
        f"rec-{i}": {
            "selected_problem_id": rows[i]["action"],
            "candidates": [
                {"problem_id": a,
                 "estimated_selection_probability": logging_probs[a]}
                for a in actions
            ],
        }
        for i in range(n)
    }
    return rows, recommendations, truth


def _check_matches_truth(result, truth):
    assert result["estimable"] is True, result["reasons"]
    for name in ("ips", "snips", "doubly_robust"):
        est = result[name]
        assert est["value"] is not None
        bound = Z_BOUND * est["standard_error"] + SMOOTHING_ALLOWANCE
        assert abs(est["value"] - truth) <= bound, (
            f"{name} {est['value']:.4f} misses truth {truth:.4f} "
            f"by more than {bound:.4f}")


def test_stage3_grind_bandit_19_uniform_logging_shifted_target():
    rows, recs, truth = _simulate(
        {"A": 0.5, "B": 0.3, "C": 0.2},
        {"A": 0.2, "B": 0.3, "C": 0.5}, seed=7)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.2, "B": 0.3,
                                               "C": 0.5},
                         recommendations=recs)
    _check_matches_truth(result, truth)


def test_stage3_grind_bandit_19_epsilon_greedy_logging_near_deterministic_target():
    rows, recs, truth = _simulate(
        {"A": 0.8, "B": 0.1, "C": 0.1},
        {"A": 0.9, "B": 0.05, "C": 0.05}, seed=11)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.9, "B": 0.05,
                                               "C": 0.05},
                         recommendations=recs)
    _check_matches_truth(result, truth)


def test_stage3_grind_bandit_19_on_policy_weights_are_one():
    rows, recs, truth = _simulate(
        {"A": 0.25, "B": 0.35, "C": 0.4},
        {"A": 0.25, "B": 0.35, "C": 0.4}, seed=13)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.25, "B": 0.35,
                                               "C": 0.4},
                         recommendations=recs)
    _check_matches_truth(result, truth)
    diag = result["diagnostics"]
    assert diag["importance_weights"]["max"] == 1.0
    assert diag["effective_sample_size_fraction"] == 1.0


def test_stage3_grind_bandit_19_zero_support_target_action_not_identifiable():
    # The target policy puts 0.3 on C, but the logging policy never
    # serves C: no row carries it and its logging probability is 0.
    rows, recs, _ = _simulate(
        {"A": 0.6, "B": 0.4},
        {"A": 0.5, "B": 0.5}, seed=7)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.35, "B": 0.35,
                                               "C": 0.3},
                         recommendations=recs)
    assert result["estimable"] is False
    assert result["ips"] is None
    assert result["snips"] is None
    assert result["doubly_robust"] is None
    assert any("support" in r for r in result["reasons"])


def test_stage3_grind_bandit_19_missing_action_zero_logging_probability():
    # A row was taken under an action whose logged propensity is 0:
    # the action was never actually served under the stated logging
    # policy, so the weight is undefined and nothing is estimated.
    rows, recs, _ = _simulate(
        {"A": 0.5, "B": 0.3, "C": 0.2},
        {"A": 0.2, "B": 0.3, "C": 0.5}, n=500, seed=7)
    rows[0] = dict(rows[0], logging_probability=0.0)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.2, "B": 0.3,
                                               "C": 0.5},
                         recommendations=recs)
    assert result["estimable"] is False
    assert result["ips"] is None
    assert any("propensity" in r for r in result["reasons"])


def test_stage3_grind_bandit_19_low_support_collapses_effective_sample_size():
    # The target policy is nearly deterministic on C, which the
    # logging policy serves with probability 0.01: weights explode
    # and the effective sample size collapses below the gate.
    rows, recs, _ = _simulate(
        {"A": 0.495, "B": 0.495, "C": 0.01},
        {"A": 0.01, "B": 0.01, "C": 0.98}, n=5000, seed=7)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.01, "B": 0.01,
                                               "C": 0.98},
                         recommendations=recs)
    assert result["estimable"] is False
    assert result["ips"] is None
    diag = result["diagnostics"]
    assert diag["effective_sample_size_fraction"] < od.MIN_ESS_FRACTION
    assert any("effective sample size" in r for r in result["reasons"])


def test_stage3_grind_bandit_19_unlinked_outcome_not_estimable():
    rows, recs, _ = _simulate(
        {"A": 0.5, "B": 0.3, "C": 0.2},
        {"A": 0.2, "B": 0.3, "C": 0.5}, n=500, seed=7)
    rows[3] = dict(rows[3], recommendation_id=None, reward=None)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.2, "B": 0.3,
                                               "C": 0.5},
                         recommendations=recs)
    assert result["estimable"] is False
    assert result["ips"] is None
    assert any("linkage" in r or "outcome" in r
               for r in result["reasons"])


def test_stage3_grind_bandit_19_unknown_recommendation_breaks_linkage():
    rows, recs, _ = _simulate(
        {"A": 0.5, "B": 0.3, "C": 0.2},
        {"A": 0.2, "B": 0.3, "C": 0.5}, n=500, seed=7)
    rows[5] = dict(rows[5], recommendation_id="rec-does-not-exist")
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.2, "B": 0.3,
                                               "C": 0.5},
                         recommendations=recs)
    assert result["estimable"] is False
    assert any("linkage" in r for r in result["reasons"])


def test_stage3_grind_bandit_19_logged_action_must_match_recommendation():
    rows, recs, _ = _simulate(
        {"A": 0.5, "B": 0.3, "C": 0.2},
        {"A": 0.2, "B": 0.3, "C": 0.5}, n=500, seed=7)
    recs = dict(recs)
    recs["rec-5"] = dict(recs["rec-5"], selected_problem_id="B")
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.2, "B": 0.3,
                                               "C": 0.5},
                         recommendations=recs)
    assert result["estimable"] is False
    assert any("linkage" in r for r in result["reasons"])


def test_stage3_grind_bandit_19_propensity_mc_error_must_be_bounded():
    # A logged propensity of 0.01 from only 100 Monte Carlo samples
    # has a worst-case standard error of 0.05, five times the
    # propensity itself: the estimate behind the log is not known
    # accurately enough to trust the weight.
    rows, recs, _ = _simulate(
        {"A": 0.5, "B": 0.3, "C": 0.2},
        {"A": 0.2, "B": 0.3, "C": 0.5}, n=500, seed=7)
    rows[0] = dict(rows[0], logging_probability=0.01,
                   logging_probability_samples=100)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.2, "B": 0.3,
                                               "C": 0.5},
                         recommendations=recs)
    assert result["estimable"] is False
    assert any("Monte Carlo" in r for r in result["reasons"])


def test_stage3_grind_bandit_19_outcome_before_exposure_is_leakage():
    rows, recs, _ = _simulate(
        {"A": 0.5, "B": 0.3, "C": 0.2},
        {"A": 0.2, "B": 0.3, "C": 0.5}, n=500, seed=7)
    rows[0] = dict(rows[0], outcome_ts=NOW - 100.0)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.2, "B": 0.3,
                                               "C": 0.5},
                         recommendations=recs)
    assert result["estimable"] is False
    assert any("leakage" in r for r in result["reasons"])


def test_stage3_grind_bandit_19_diagnostics_report_weight_percentiles():
    rows, recs, _ = _simulate(
        {"A": 0.5, "B": 0.3, "C": 0.2},
        {"A": 0.2, "B": 0.3, "C": 0.5}, n=2000, seed=7)
    diag = od.diagnose(rows, target_actions={"A", "B", "C"},
                       recommendations=recs)
    assert diag["estimable"] is True
    weights = diag["importance_weights"]
    assert 0.4 <= weights["p50"] <= 1.0
    assert weights["p99"] == 2.5
    assert weights["max"] == 2.5
    assert weights["fraction_above_cap"] == 0.0
    assert diag["support"]["fraction"] == 1.0
    assert diag["linkage"]["fraction"] == 1.0
    assert diag["unique_assessment_coverage"]["fraction"] == 1.0
    assert diag["time_leakage"]["count"] == 0
    assert diag["missing_outcomes"] == 0


def test_stage3_grind_bandit_19_no_default_lift_number():
    # The gate returns no numeric policy value at all when the
    # diagnostics fail: callers cannot mistake a refusal for a lift.
    rows, recs, _ = _simulate(
        {"A": 0.6, "B": 0.4},
        {"A": 0.5, "B": 0.5}, n=100, seed=7)
    result = oe.estimate(rows, target_actions={"A", "B", "C"},
                         target_probabilities={"A": 0.35, "B": 0.35,
                                               "C": 0.3},
                         recommendations=recs)
    assert result["estimable"] is False
    assert result["ips"] is None and result["snips"] is None
    assert result["doubly_robust"] is None
    assert result["sensitivity"] is None
    assert result["reasons"]
