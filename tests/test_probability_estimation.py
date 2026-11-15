"""Selection-probability estimator: analytic oracle and invariance (GB12).

For two fixed candidate contexts x1, x2 and posterior theta ~ N(mu, Sigma),
with d = x1 - x2, the exact probability of picking candidate 1 is
Phi(mu^T d / sqrt(d^T Sigma d)) whenever the variance is strictly
positive, ignoring zero-probability continuous ties. These tests pin the
Monte Carlo estimator in bandit.estimate_selection_probabilities against
that closed form, with the tolerance justified from the Monte Carlo
standard error and the add-one smoothing shift accounted for as its own
term, and pin that the estimator can never change the pick.
"""

import math

import numpy as np
import pytest

from grindbandit import bandit


def _phi(z: float) -> float:
    """Standard normal CDF, computed independently of the implementation."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _state(mean, prec):
    return {"bandit": {"mean": list(mean), "prec": np.asarray(prec, dtype=float).tolist()}}


def _problem(pid, topics=("dp",), difficulty="Medium"):
    return {"id": pid, "title": f"P{pid}", "difficulty": difficulty, "topics": list(topics)}


def _two_problems():
    return [_problem(1, ["dp"]), _problem(2, ["graphs"])]


# Two fixed candidate contexts used by the oracle tests. They are plain
# vectors, not built through context(), so the oracle comparison isolates
# the estimator from the feature layout.
X1 = np.array([0.8, 0.6, 0.9, 0.5, 1.0])
X2 = np.array([0.3, 0.4, 0.7, 0.2, 1.0])


def test_stage3_grind_bandit_12_oracle_known_answers():
    # Zero mean: the two candidates are equally likely whatever d is.
    mu = np.zeros(5)
    cov = np.eye(5)
    assert bandit.analytic_pair_selection_probability(mu, cov, X1, X2) == pytest.approx(0.5)

    # z = +1 exactly: d = e0, mu = e0, cov = I gives mu^T d = 1, var = 1.
    e0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    zero = np.zeros(5)
    assert bandit.analytic_pair_selection_probability(mu + e0, cov, e0, zero) == pytest.approx(
        _phi(1.0), abs=1e-12
    )
    # z = -1 exactly.
    assert bandit.analytic_pair_selection_probability(mu - e0, cov, e0, zero) == pytest.approx(
        _phi(-1.0), abs=1e-12
    )
    # Correlated direction: d = (1, 1, 0, 0, 0), mu = e0, cov = I gives
    # z = 1 / sqrt(2).
    d = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    assert bandit.analytic_pair_selection_probability(e0, cov, d, zero) == pytest.approx(
        _phi(1.0 / math.sqrt(2.0)), abs=1e-12
    )


def test_stage3_grind_bandit_12_oracle_matches_monte_carlo():
    mean = [0.6, -0.4, 0.2, 0.0, 0.1]
    prec = (np.eye(5) * 2.0).tolist()
    state = _state(mean, prec)
    cov = np.linalg.inv(np.array(prec))
    oracle = bandit.analytic_pair_selection_probability(
        np.array(mean), cov, X1, X2
    )
    assert 0.0 < oracle < 1.0

    samples = 20_000
    probs, ses = bandit.estimate_selection_probabilities_with_error(
        state, [X1, X2], seed=42, samples=samples
    )
    assert abs(sum(probs) - 1.0) < 1e-12
    assert all(0.0 < p < 1.0 for p in probs)
    assert all(se >= 0.0 for se in ses)

    # Tolerance, justified from the Monte Carlo standard error: the raw
    # win frequency is a binomial proportion, so five standard errors is
    # the simulation allowance, and the add-one smoothing shift is at
    # most 1 / (samples + n_candidates), kept as its own term.
    smoothing_bound = 1.0 / (samples + 2)
    assert abs(probs[0] - oracle) <= 5.0 * ses[0] + smoothing_bound
    assert abs(probs[1] - (1.0 - oracle)) <= 5.0 * ses[1] + smoothing_bound

    # The reported error is consistent with the binomial standard error
    # of the smoothed estimate at this sample count.
    approx_se = math.sqrt(probs[0] * (1.0 - probs[0]) / samples)
    assert ses[0] == pytest.approx(approx_se, rel=0.05)


def test_stage3_grind_bandit_12_smoothing_accounted_explicitly():
    # Identical contexts: argmax always takes the first candidate, so the
    # raw win counts are exactly [n, 0] and the smoothed values are exact.
    state = _state([0.0] * 5, np.eye(5).tolist())
    samples = 10
    probs, ses = bandit.estimate_selection_probabilities_with_error(
        state, [X1, X1], seed=3, samples=samples
    )
    assert probs == pytest.approx([11.0 / 12.0, 1.0 / 12.0])
    assert abs(sum(probs) - 1.0) < 1e-12
    assert all(0.0 < p < 1.0 for p in probs)
    # A candidate with zero raw wins still gets a positive smoothed
    # probability, and its simulation error is zero because the raw
    # frequency is pinned at 0 by the tie.
    assert ses == pytest.approx([0.0, 0.0])
    # The plain estimator returns exactly the same smoothed values.
    assert bandit.estimate_selection_probabilities(
        state, [X1, X1], seed=3, samples=samples
    ) == pytest.approx(probs)


def test_stage3_grind_bandit_12_degenerate_cases_stable():
    cov0 = np.zeros((5, 5))
    e0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    zero = np.zeros(5)
    # Zero variance: the draw is the mean, so the sign of mu^T d decides.
    assert bandit.analytic_pair_selection_probability(e0, cov0, e0, zero) == 1.0
    assert bandit.analytic_pair_selection_probability(-e0, cov0, e0, zero) == 0.0
    # Zero variance and zero mean difference: an exact tie, reported 0.5.
    assert bandit.analytic_pair_selection_probability(zero, cov0, e0, zero) == 0.5
    # Identical contexts under any posterior: d = 0, so the oracle is 0.5.
    assert bandit.analytic_pair_selection_probability(
        np.array([0.6, -0.4, 0.2, 0.0, 0.1]), np.eye(5), X1, X1
    ) == pytest.approx(0.5)

    # Monte Carlo stays finite and sums to 1 on identical contexts.
    state = _state([0.0] * 5, np.eye(5).tolist())
    probs = bandit.estimate_selection_probabilities(state, [X1, X1], seed=9, samples=100)
    assert all(math.isfinite(p) for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-12

    # Near-zero posterior variance: the pick is effectively deterministic.
    state = _state([1.0, 0.0, 0.0, 0.0, 0.0], (np.eye(5) * 1e12).tolist())
    probs = bandit.estimate_selection_probabilities(
        state, [e0, zero], seed=9, samples=1000
    )
    assert all(math.isfinite(p) for p in probs)
    assert probs[0] > 0.99


def test_stage3_grind_bandit_12_paired_selection_mocked_estimation(monkeypatch):
    state = {}
    now = 1_000_000.0
    plain = bandit.build_recommendation(state, _two_problems(), now, seed=7)
    assert plain["probability_estimate"]["method"] == (
        "monte_carlo_posterior_resampling_add_one_smoothed"
    )

    def _fixed(state, contexts, seed, samples=bandit.PROBABILITY_SAMPLES):
        return [0.99, 0.01]

    monkeypatch.setattr(bandit, "estimate_selection_probabilities", _fixed)
    mocked = bandit.build_recommendation(state, _two_problems(), now, seed=7)
    assert mocked["selected_problem_id"] == plain["selected_problem_id"]
    assert [c["estimated_selection_probability"] for c in mocked["candidates"]] == [0.99, 0.01]

    def _boom(state, contexts, seed, samples=bandit.PROBABILITY_SAMPLES):
        raise RuntimeError("estimation must never decide the pick")

    monkeypatch.setattr(bandit, "estimate_selection_probabilities", _boom)
    with pytest.raises(RuntimeError):
        bandit.build_recommendation(state, _two_problems(), now, seed=7)


def test_stage3_grind_bandit_12_changed_sample_counts_same_pick():
    state = {}
    now = 1_000_000.0
    picks = {
        samples: bandit.build_recommendation(
            state, _two_problems(), now, seed=7, samples=samples
        )["selected_problem_id"]
        for samples in (100, 1_000, 50_000)
    }
    assert len(set(picks.values())) == 1


def test_stage3_grind_bandit_12_estimation_uses_separate_rng():
    state = _state([0.6, -0.4, 0.2, 0.0, 0.1], (np.eye(5) * 2.0).tolist())
    before = bandit.sample_theta(state, seed=7)
    bandit.estimate_selection_probabilities(state, [X1, X2], seed=8, samples=1000)
    after = bandit.sample_theta(state, seed=7)
    assert np.array_equal(before, after)

    first = bandit.estimate_selection_probabilities(state, [X1, X2], seed=8, samples=1000)
    second = bandit.estimate_selection_probabilities(state, [X1, X2], seed=8, samples=1000)
    assert first == second


def test_stage3_grind_bandit_12_standard_error_scales_with_samples():
    # Zero mean gives a win rate near 0.5 at any sample count, so the
    # standard error should shrink close to sqrt(n) scaling.
    state = _state([0.0] * 5, np.eye(5).tolist())
    _, se_small = bandit.estimate_selection_probabilities_with_error(
        state, [X1, X2], seed=11, samples=1_000
    )
    _, se_large = bandit.estimate_selection_probabilities_with_error(
        state, [X1, X2], seed=11, samples=40_000
    )
    assert se_large[0] < se_small[0] / 4.0


def test_stage3_grind_bandit_12_old_replay_unaffected():
    state = {}
    now = 1_000_000.0
    rec = bandit.build_recommendation(state, _two_problems(), now, seed=5)
    assert bandit.replay_selection(rec) == rec["selected_problem_id"]

    # A hand-built record in the pre-existing shape replays the same way.
    mean = np.array(rec["posterior"]["mean"])
    prec = np.array(rec["posterior"]["prec"])
    snapshot = {"bandit": {"mean": mean.tolist(), "prec": prec.tolist()}}
    theta = bandit.sample_theta(snapshot, seed=rec["seed"])
    scores = [
        bandit.score(theta, np.asarray(c["context"], dtype=float))
        for c in rec["candidates"]
    ]
    expected = rec["candidates"][int(np.argmax(scores))]["problem_id"]
    assert bandit.replay_selection(rec) == expected

    # The estimator disclosure is unchanged by this unit.
    est = rec["probability_estimate"]
    assert est["method"] == "monte_carlo_posterior_resampling_add_one_smoothed"
    assert est["seed"] == (rec["seed"] + 1) % (2**63)
    assert est["samples"] == bandit.PROBABILITY_SAMPLES
