"""Stage 3 GB15: sparse-data cold-start and ability-difficulty separation.

Gate tests for the shadow skill model. The production mastery model
(topic means with a fixed 30-day decay) is never touched: the learned
prediction heads live in a separate shadow registry and are adopted
only when they beat all three baselines (global solve rate, existing
topic means, deterministic per-difficulty rate heuristic) out of
sample on Brier and log loss, with a bootstrap interval against the
production topic means that excludes zero, and with fitted parameters
inside sane bounds. Otherwise the simple model stays.

Real practice data in this repo is too sparse to identify anything
(the scripted demo history holds 30 attempts against a predeclared
60 train plus 20 holdout minimum), so the gate verdict on real data
is NOT IDENTIFIABLE and the simple model stays. Controlled simulation
with a known ground truth validates that the machinery adopts a
genuinely better head and rejects one that is not better: with both
topic ability and difficulty in play the Rasch-style head adopts,
while with only topic signal, only difficulty signal, or no signal
at all it correctly keeps the simple model.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest

from grindbandit import mastery
from grindbandit.research import evaluate_skill as ev
from grindbandit.research import skill_model as sm

# Topic abilities used in the simulation worlds: strong on array and
# string, weak on dp, graph, and backtracking, mirroring the scripted
# demo history's topic strengths.
TOPIC_ABILITY = {"array": 1.2, "string": 1.2, "dp": -1.2,
                 "graph": -1.2, "backtracking": -1.2}
DIFFICULTY_OFFSET = {"Easy": -1.0, "Medium": 0.0, "Hard": 1.0}
# The same strengths as solve rates, for the rates-family worlds.
TOPIC_RATE = {"array": 0.7, "string": 0.7, "dp": 0.3,
              "graph": 0.3, "backtracking": 0.3}
NEUTRAL_DIFFICULTY_RATE = {"Easy": 0.5, "Medium": 0.5, "Hard": 0.5}


def _report(truth, n=400, seed=7):
    obs = sm.simulate_observations(n, truth=truth, seed=seed)
    train, holdout = sm.split_holdout(obs)
    return sm.compare_models(train, holdout), train, holdout


def test_stage3_grind_bandit_15_topic_world_keeps_simple():
    """Only topic signal, at moderate strength, and no difficulty
    signal: neither learned head robustly beats the baselines. The
    shrunk head does not beat the raw topic means out of sample, and
    the Rasch head's edge over the production topic means stays inside
    its bootstrap interval. The simple model stays."""
    report, _, _ = _report({"topic_rate": TOPIC_RATE,
                            "difficulty_rate": NEUTRAL_DIFFICULTY_RATE})
    assert report["verdict"] == "KEEP_SIMPLE", report
    assert report["gate_failures"]["hierarchical_beta"], report["gate_failures"]
    assert report["gate_failures"]["rasch"], report["gate_failures"]
    assert report["models"]["rasch"]["brier_vs_production"]["ci_high"] >= 0.0


def test_stage3_grind_bandit_15_both_world_adopts_rasch():
    """Topic ability and difficulty both matter: the Rasch-style head,
    which separates the two, must beat every baseline. Hard problems
    are rare in the bank (10 of 218) and confounded with dp, so the
    world needs 2,000 attempts before the separation is identifiable;
    at that size the adoption is stable across seeds 7, 11, and 13."""
    report, _, _ = _report({"ability": 0.0, "topic_ability": TOPIC_ABILITY,
                            "difficulty_offset": DIFFICULTY_OFFSET},
                           n=2000)
    assert report["verdict"] == "ADOPT:rasch", report
    learned = report["models"]["rasch"]
    for kind in ("global_rate", "topic_mean", "difficulty_heuristic"):
        base = report["models"][kind]
        assert learned["brier"] < base["brier"]
        assert learned["log_loss"] < base["log_loss"]
    assert learned["brier_vs_production"]["ci_high"] < 0.0
    assert learned["pathological"] == []


def test_stage3_grind_bandit_15_null_world_keeps_simple():
    """No topic or difficulty signal at all (every attempt a coin
    flip): no learned head may win, and the simple model stays."""
    report, _, _ = _report({"topic_rate": {},
                            "difficulty_rate": NEUTRAL_DIFFICULTY_RATE})
    assert report["verdict"] == "KEEP_SIMPLE", report


def test_stage3_grind_bandit_15_difficulty_only_world_keeps_simple():
    """Only difficulty matters: the Rasch-style head carries the same
    information as the deterministic difficulty heuristic, so it must
    not claim a robust win over it."""
    report, _, _ = _report({"ability": 0.0, "topic_ability": {},
                            "difficulty_offset": DIFFICULTY_OFFSET})
    assert report["verdict"] == "KEEP_SIMPLE", report
    assert "rasch" in report["gate_failures"]
    assert report["gate_failures"]["rasch"], report["gate_failures"]


def test_stage3_grind_bandit_15_demo_state_not_identifiable():
    """The scripted demo history holds 30 attempts against the 60
    train plus 20 holdout minimum."""
    from common import build_demo_state

    state = build_demo_state(42)
    report = ev.evaluate(state)
    assert report["verdict"] == "NOT_IDENTIFIABLE", report
    assert report["n_observations"] == 30
    assert report["n_attempts"] == 30
    assert report["production_model"] == "topic_mean"


def test_stage3_grind_bandit_15_empty_state_not_identifiable():
    report = ev.evaluate({"log": []})
    assert report["verdict"] == "NOT_IDENTIFIABLE"
    assert report["n_observations"] == 0


def test_stage3_grind_bandit_15_all_solved_train_is_pathological():
    """A train set with no failures pushes the Rasch ability to its
    bound. That is a pathological extreme, never an adoption."""
    obs = [{"problem_id": 1, "difficulty": "Easy", "topics": ["array"],
            "solved": True, "ts": 1_700_000_000.0 + i * 86400.0,
            "attempt_index": i, "prior_attempts_on_problem": i,
            "repeat": i > 0}
           for i in range(80)]
    train, holdout = sm.split_holdout(obs)
    report = sm.compare_models(train, holdout)
    assert report["models"]["rasch"]["pathological"], report["models"]["rasch"]
    assert report["verdict"] in ("KEEP_SIMPLE", "NOT_IDENTIFIABLE")


def test_stage3_grind_bandit_15_holdout_brier_matches_plain_math_oracle():
    """Independent oracle: recompute the global-rate holdout Brier
    from plain math and check it against the module's number."""
    report, _, holdout = _report({"ability": 0.0,
                                  "topic_ability": TOPIC_ABILITY,
                                  "difficulty_offset": DIFFICULTY_OFFSET})
    entry = report["models"]["global_rate"]
    rate = entry["rate"]
    oracle = sum((rate - (1.0 if ob["solved"] else 0.0)) ** 2
                 for ob in holdout) / len(holdout)
    assert entry["brier"] == pytest.approx(oracle)


def test_stage3_grind_bandit_15_production_mastery_untouched():
    """The gate never rewrites live mastery: evaluating the shadow
    model changes no mastery value, and the topic-mean head stays
    the production prediction."""
    from common import build_demo_state

    state = build_demo_state(42)
    before = {t: mastery.mean(state, t) for t in state["topics"]}
    ev.evaluate(state)
    after = {t: mastery.mean(state, t) for t in state["topics"]}
    assert before == after
    assert mastery.HALF_LIFE_DAYS == 30.0
    assert sm.active_production_model() == "topic_mean"
    assert sm.REGISTRY["hierarchical_beta"]["status"] == "shadow"
    assert sm.REGISTRY["rasch"]["status"] == "shadow"


def test_stage3_grind_bandit_15_breakdowns_reported_per_topic_and_difficulty():
    """Every scored model reports per-topic and by-difficulty holdout
    breakdowns, so a win cannot hide a failure on Hard problems."""
    report, _, _ = _report({"ability": 0.0, "topic_ability": TOPIC_ABILITY,
                            "difficulty_offset": DIFFICULTY_OFFSET})
    for kind, entry in report["models"].items():
        by_diff = {r["difficulty"] for r in entry["breakdown_by_difficulty"]}
        assert {"Easy", "Medium", "Hard"} <= by_diff, kind
        hard = next(r for r in entry["breakdown_by_difficulty"]
                    if r["difficulty"] == "Hard")
        assert hard["n"] > 0
        assert entry["breakdown_by_topic"], kind
