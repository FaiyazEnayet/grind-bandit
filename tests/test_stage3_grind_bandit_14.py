"""Stage 3 GB14: adaptive memory model, learnable half-life versus fixed decay.

Gate tests for the shadow retention model. The production mastery decay
(fixed 30 days) is never touched: the learned models live in a separate
shadow registry and are adopted only when they beat both the trivial
global rate and the fixed 30-day baseline out of sample, with a bootstrap
interval that excludes zero, and with fitted half-lives inside sane
bounds. Otherwise the simple model stays.

Real practice data in this repo is too sparse to identify anything (the
scripted demo history yields 40 retention observations against a
predeclared 60 train plus 20 holdout minimum), so the gate verdict on
real data is NOT IDENTIFIABLE and the fixed model stays. Controlled
simulation with a known ground-truth half-life validates that the
machinery adopts a genuinely better half-life and rejects one that is
not better.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from grindbandit import mastery
from grindbandit.research import evaluate_retention as ev
from grindbandit.research import retention as rt


def test_stage3_grind_bandit_14_simulation_adopts_better_half_life():
    """Known truth h=12 days: the learned model must beat fixed 30 out of sample."""
    obs = rt.simulate_observations(400, half_lives=12.0, seed=7)
    train, holdout = rt.split_holdout(obs)
    report = rt.compare_models(train, holdout)
    assert report["verdict"] == "ADOPT:global_h", report
    learned = report["models"]["global_h"]
    fixed = report["models"]["fixed30"]
    assert learned["brier"] < fixed["brier"]
    assert learned["log_loss"] < fixed["log_loss"]
    assert learned["brier_vs_fixed30"]["ci_high"] < 0.0
    assert abs(learned["half_life"] - 12.0) < 3.0


def test_stage3_grind_bandit_14_simulation_rejects_equal_half_life():
    """Known truth h=30 days: the learned model must not beat the fixed baseline."""
    obs = rt.simulate_observations(400, half_lives=30.0, seed=7)
    train, holdout = rt.split_holdout(obs)
    report = rt.compare_models(train, holdout)
    assert report["verdict"] == "KEEP_SIMPLE", report
    assert report["models"]["global_h"]["half_life"] is not None


def test_stage3_grind_bandit_14_simulation_topic_model_beats_global():
    """Topics with genuinely different half-lives (8 vs 40 days): the
    shrunk topic model must beat the single global half-life out of sample."""
    obs = rt.simulate_observations(
        600, half_lives={"array": 8.0, "dp": 40.0}, seed=11)
    train, holdout = rt.split_holdout(obs)
    report = rt.compare_models(train, holdout)
    topic = report["models"]["topic_h"]
    glob = report["models"]["global_h"]
    assert topic["brier"] < glob["brier"]
    assert topic["log_loss"] < glob["log_loss"]


def test_stage3_grind_bandit_14_demo_state_not_identifiable():
    """The scripted demo history is too small for the predeclared gate:
    40 observations against 60 train plus 20 holdout required."""
    from common import build_demo_state

    state = build_demo_state(42)
    report = ev.evaluate(state)
    assert report["verdict"] == "NOT_IDENTIFIABLE", report
    assert report["n_observations"] == 40
    assert report["production_model"] == "fixed30"


def test_stage3_grind_bandit_14_empty_state_not_identifiable():
    report = ev.evaluate({"log": []})
    assert report["verdict"] == "NOT_IDENTIFIABLE"
    assert report["n_observations"] == 0


def test_stage3_grind_bandit_14_all_solved_train_is_pathological():
    """A train set with no failures pushes the fitted half-life to the
    upper bound. That is a pathological extreme, never an adoption."""
    obs = [
        {"topic": "array", "delta_days": 1.0 + i * 0.1, "solved": True,
         "prior_successes": 1, "ts": 1_700_000_000.0 + i * 86400.0,
         "last_ts": 1_700_000_000.0 + i * 86400.0 - 86400.0}
        for i in range(80)
    ]
    train, holdout = rt.split_holdout(obs)
    report = rt.compare_models(train, holdout)
    assert report["models"]["global_h"]["at_bound"] is True
    assert report["verdict"] in ("KEEP_SIMPLE", "NOT_IDENTIFIABLE")


def test_stage3_grind_bandit_14_production_decay_untouched():
    """The gate never rewrites live mastery: evaluating the shadow model
    changes no mastery value, and the fixed 30-day half-life still rules."""
    from common import build_demo_state

    state = build_demo_state(42)
    before = {t: mastery.mean(state, t) for t in state["topics"]}
    ev.evaluate(state)
    after = {t: mastery.mean(state, t) for t in state["topics"]}
    assert before == after
    assert mastery.HALF_LIFE_DAYS == 30.0
    assert rt.FIXED_HALF_LIFE_DAYS == mastery.HALF_LIFE_DAYS
    assert rt.active_production_model() == "fixed30"
    assert rt.REGISTRY["global_h"]["status"] == "shadow"
    assert rt.REGISTRY["topic_h"]["status"] == "shadow"
    assert rt.REGISTRY["recurrence_h"]["status"] == "shadow"
