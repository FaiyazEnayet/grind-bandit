"""Stage 3 GB4: explanations must match the actual sampled draw.

The old explain() derived its driver claim from the posterior mean while
selection used a sampled theta, so an explanation could credit a factor
that did not drive the realized draw. These tests pin the faithful form:
per-feature sampled contributions that sum to the logged sampled score,
a driver equal to the largest sampled contribution, an exploration label
on wide-prior picks, and uncertainty reported as a model estimate with
its resampling seed recorded, never as observed proficiency.
"""

import numpy as np
import pytest

from grindbandit import bandit

DAY = 86400.0


def _problem(pid, topics, difficulty="Medium"):
    return {"id": pid, "title": f"P{pid}", "difficulty": difficulty, "topics": topics}


def _candidates():
    return [
        _problem(1, ["dp", "array"]),
        _problem(2, ["graph"]),
        _problem(3, ["string", "array"], "Easy"),
    ]


def _trained_state():
    """Posterior with enough diverse observations to concentrate."""
    s = {}
    rng = np.random.default_rng(7)
    for _ in range(20):
        x = np.array([rng.random(), rng.random(), rng.random(), rng.random(), 1.0])
        bandit.observe(s, x, 0.3)
    return s


def test_explanation_contributions_sum_to_sampled_score_20_seeds():
    for seed in range(20):
        s = _trained_state()
        rec = bandit.build_recommendation(s, _candidates(), 1000.0, seed=seed)
        expl = rec["explanation"]
        total = sum(expl["contributions"].values()) + expl["intercept"]
        assert abs(total - expl["sampled_score"]) < 1e-10
        selected = [c for c in rec["candidates"] if c["selected"]][0]
        assert abs(expl["sampled_score"] - selected["sampled_score"]) < 1e-10
        assert bandit.replay_selection(rec) == rec["selected_problem_id"]


def test_explanation_driver_is_largest_sampled_contribution():
    # A theta that values recency far above every other factor, on a
    # problem never practiced (recency 1.0). The driver must be recency,
    # whatever the posterior mean says.
    s = {}
    theta = np.array([0.1, 0.1, 0.1, 5.0, 0.0])
    expl = bandit.explain_sampled(
        _problem(1, ["dp"]), s, 1000.0, theta, uncertainty_seed=99
    )
    assert expl["driver"] == "recency"
    assert expl["driver_contribution"] == pytest.approx(theta[3] * 1.0)
    assert any("stale" in line for line in expl["lines"])
    total = sum(expl["contributions"].values()) + expl["intercept"]
    assert abs(total - expl["sampled_score"]) < 1e-10


def test_wide_prior_pick_is_labeled_exploration():
    rec = bandit.build_recommendation({}, _candidates(), 1000.0, seed=3)
    expl = rec["explanation"]
    assert expl["exploration"] is True
    assert any("exploration" in line for line in expl["lines"])


def test_concentrated_posterior_pick_is_not_labeled_exploration():
    rec = bandit.build_recommendation(_trained_state(), _candidates(), 1000.0, seed=3)
    assert rec["explanation"]["exploration"] is False


def test_uncertainty_uses_at_least_1000_draws_with_recorded_seed():
    s = _trained_state()
    rec = bandit.build_recommendation(s, _candidates(), 1000.0, seed=3)
    unc = rec["explanation"]["uncertainty"]
    assert unc["samples"] >= 1000
    assert isinstance(unc["seed"], int)
    # Recompute from the logged posterior snapshot and recorded seed.
    mean = np.array(rec["posterior"]["mean"])
    cov = np.linalg.inv(np.array(rec["posterior"]["prec"]))
    rng = np.random.default_rng(unc["seed"])
    thetas = rng.multivariate_normal(mean, cov, size=unc["samples"])
    selected = [c for c in rec["candidates"] if c["selected"]][0]
    scores = thetas @ np.array(selected["context"], dtype=float)
    assert abs(float(scores.std()) - unc["score_sd"]) < 1e-9


def test_uncertainty_never_presented_as_observed_proficiency():
    rec = bandit.build_recommendation({}, _candidates(), 1000.0, seed=3)
    text = " ".join(rec["explanation"]["lines"])
    assert "observed proficiency" not in text
    assert "modeled mastery" in text
    assert "model" in text


def test_next_prints_the_logged_sampled_explanation(tmp_path, monkeypatch, capsys):
    from grindbandit import cli
    from grindbandit import state as st

    monkeypatch.setenv("GRIND_BANDIT_STATE", str(tmp_path / "state.json"))
    cli.main(["next", "--seed", "11"])
    out = capsys.readouterr().out
    rec = st.load()["recommendations"][0]
    for line in rec["explanation"]["lines"]:
        assert line in out


def test_explain_lines_match_sampled_theta_not_posterior_mean():
    # Posterior mean favors gap_min, sampled theta favors recency.
    s = {"bandit": {"mean": [2.0, 0.0, 0.0, 0.0, 0.0],
                    "prec": np.eye(bandit.D).tolist()}}
    theta = np.array([0.1, 0.1, 0.1, 5.0, 0.0])
    lines = bandit.explain(_problem(1, ["dp"]), s, 1000.0, theta,
                           uncertainty_seed=99)
    assert any("stale" in line for line in lines)
    assert not any("biggest gap" in line for line in lines)
