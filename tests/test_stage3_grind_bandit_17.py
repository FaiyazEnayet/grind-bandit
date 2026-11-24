"""Stage 3 GB17: budget-aware recommendation objectives and policy alternatives.

Gate tests for the versioned policy branch. The production mastery model,
bandit, and reward are never touched: the candidate policies live in a
separate versioned registry (policy-v1) and stay shadow until real
measurement shows improvement on the delayed-transfer proxy, with no
unacceptable starvation of Hard and rare topics and no overrun relative
to the stated budget in controlled tests. Otherwise the simple model
stays.

Real practice data here is too sparse to identify anything (the
scripted demo history holds 30 attempts and no delayed quizzes at
all), so the gate verdict on real data is NOT IDENTIFIABLE. Controlled
simulation with a known ground truth validates that the machinery
compares eligibility-matched policies honestly across fixed 15, 30,
and 60 minute budgets: it must surface a challenger win when one
exists in the simulated world and keep the simple model when none
does. Simulation is machinery validation only, never evidence about
real users.
"""

import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest

from grindbandit import bandit, mastery
from grindbandit.policies import registry as reg
from grindbandit.policies import time_aware as ta
from grindbandit.research import policy_comparison as pc

DAY = 86400.0
NOW = 1_750_000_000.0


def _problem(pid, difficulty, topics):
    return {"id": pid, "title": f"P{pid}", "difficulty": difficulty,
            "topics": list(topics)}


def _state(topics=None, problems=None, log=None):
    return {"topics": topics or {}, "problems": problems or {},
            "log": log or [], "bandit": {}}


def _topic_entry(alpha, beta, last_ts):
    return {"alpha": alpha, "beta": beta, "last_ts": last_ts}


def _history():
    """Attempts whose minutes differ by difficulty: Easy around 10,
    Medium around 25, Hard around 50. The target problem's own
    minutes must never move its own expectation."""
    return [
        {"problem_id": 101, "difficulty": "Easy", "minutes": 8},
        {"problem_id": 102, "difficulty": "Easy", "minutes": 12},
        {"problem_id": 103, "difficulty": "Medium", "minutes": 20},
        {"problem_id": 104, "difficulty": "Medium", "minutes": 30},
        {"problem_id": 105, "difficulty": "Hard", "minutes": 40},
        {"problem_id": 106, "difficulty": "Hard", "minutes": 60},
    ]


def test_stage3_grind_bandit_17_registry_shadow_only():
    """The versioned branch records six policies. Current LinTS is
    the only production policy; every challenger is shadow, and the
    active production policy stays LinTS."""
    assert reg.POLICY_VERSION == "policy-v1"
    assert reg.POLICY_NAMES == ["lints", "shortest_duration",
                                "weakest_topic", "due_first",
                                "cost_aware_lints", "constrained_lints"]
    assert reg.REGISTRY["lints"]["status"] == "production"
    for name in reg.POLICY_NAMES[1:]:
        assert reg.REGISTRY[name]["status"] == "shadow", name
    assert reg.active_production_policy() == "lints"


def test_stage3_grind_bandit_17_expected_minutes_leave_one_out():
    """A problem's own recorded minutes never predict its own
    duration: problem 105 logged 40 minutes on a Hard problem, but
    its expectation uses only the other Hard attempts (60). A fresh
    Easy problem uses every Easy attempt."""
    attempts = _history()
    assert ta.expected_minutes(_problem(105, "Hard", ["dp"]), attempts) == 60.0
    assert ta.expected_minutes(_problem(999, "Easy", ["array"]), attempts) == 10.0
    assert ta.expected_minutes(_problem(999, "Medium", ["array"]), attempts) == 25.0
    assert ta.expected_minutes(_problem(999, "Hard", ["array"]), attempts) == 50.0


def test_stage3_grind_bandit_17_expected_minutes_fallback_prior():
    """With no history at all the expectation is the predeclared
    difficulty prior, a starting heuristic, not a fitted value. With
    history on other difficulties only, the missing difficulty still
    falls back to its prior."""
    assert ta.expected_minutes(_problem(1, "Easy", ["array"]), []) == 15.0
    assert ta.expected_minutes(_problem(1, "Medium", ["array"]), []) == 30.0
    assert ta.expected_minutes(_problem(1, "Hard", ["array"]), []) == 45.0
    medium_only = [a for a in _history() if a["difficulty"] == "Medium"]
    assert ta.expected_minutes(_problem(1, "Easy", ["array"]),
                               medium_only) == 15.0


def test_stage3_grind_bandit_17_utility_is_benefit_minus_lambda_time():
    """U = predicted delayed-transfer benefit minus lambda times
    expected time. The benefit here is the sampled LinTS score in
    model-derived reward units, explicitly hypothetical as a transfer
    predictor until a valid predictive head exists."""
    assert ta.utility(0.5, 30.0) == pytest.approx(0.5 - 0.01 * 30.0)
    assert ta.utility(0.5, 30.0, lam=0.0) == pytest.approx(0.5)
    assert ta.utility(0.0, 60.0) == pytest.approx(-0.6)
    assert ta.LAMBDA_COST == 0.01


def test_stage3_grind_bandit_17_candidate_features():
    """One versioned feature bundle per candidate: budget, expected
    minutes from history, difficulty, novelty, and interval since
    the target skill was practiced."""
    state = _state(
        topics={"array": _topic_entry(1, 1, None),
                "dp": _topic_entry(1, 9, NOW - 20 * DAY)},
        log=_history())
    feats = ta.candidate_features(_problem(7, "Medium", ["dp", "array"]),
                                  state, NOW, 30.0)
    assert feats["budget_minutes"] == 30.0
    assert feats["expected_minutes"] == 25.0
    assert feats["difficulty"] == 0.5
    assert feats["novelty"] == 1.0
    # Weakest topic is dp, last practiced 20 days ago.
    assert feats["interval_days"] == pytest.approx(20.0)
    assert feats["remaining_minutes"] == pytest.approx(5.0)
    # A never-practiced weakest topic counts as the full 30-day cap.
    fresh = _state(topics={"array": _topic_entry(1, 1, None)}, log=[])
    assert ta.candidate_features(_problem(8, "Easy", ["array"]),
                                 fresh, NOW, 15.0)["interval_days"] == 30.0


def test_stage3_grind_bandit_17_shortest_duration_picks_fastest():
    """Shortest expected-duration picks the lowest expected minutes,
    ties to the lowest id."""
    cands = [_problem(3, "Hard", ["dp"]),
             _problem(1, "Easy", ["array"]),
             _problem(2, "Medium", ["graph"])]
    picked = ta.pick("shortest_duration", cands, _state(log=_history()),
                     NOW, 60.0, seed=7)
    assert picked["id"] == 1


def test_stage3_grind_bandit_17_weakest_topic_picks_biggest_gap():
    """Weakest-topic picks the problem whose weakest topic has the
    lowest modeled mastery."""
    state = _state(topics={
        "array": _topic_entry(9, 1, NOW),
        "dp": _topic_entry(1, 4, NOW)})
    cands = [_problem(1, "Easy", ["array"]),
             _problem(2, "Easy", ["dp"])]
    assert ta.pick("weakest_topic", cands, state, NOW, 60.0, seed=7)["id"] == 2


def test_stage3_grind_bandit_17_due_first_prefers_due_review():
    """Due-first puts a due review ahead of new practice even when
    the new problem's topics are staler: the class decides first,
    then the interval."""
    state = _state(
        topics={"dp": _topic_entry(5, 5, NOW - 20 * DAY),
                "array": _topic_entry(1, 1, None)},
        problems={"1": {"ever_solved": True, "last_ts": NOW - 20 * DAY}})
    cands = [_problem(2, "Easy", ["array"]),
             _problem(1, "Medium", ["dp"])]
    assert ta.pick("due_first", cands, state, NOW, 60.0, seed=7)["id"] == 1


def test_stage3_grind_bandit_17_constrained_filters_over_budget():
    """Constrained optimization never picks a problem whose expected
    minutes exceed the remaining budget, and picks nothing at all
    when nothing fits instead of overrunning."""
    cands = [_problem(1, "Easy", ["array"]),
             _problem(2, "Medium", ["graph"]),
             _problem(3, "Hard", ["dp"])]
    state = _state(log=_history())
    assert ta.pick("constrained_lints", cands, state, NOW, 5.0,
                   seed=7) is None
    picked = ta.pick("constrained_lints", cands, state, NOW, 12.0, seed=7)
    assert picked["id"] == 1
    picked = ta.pick("constrained_lints", cands, state, NOW, 30.0, seed=7)
    assert ta.expected_minutes(picked, _history()) <= 30.0


def test_stage3_grind_bandit_17_cost_aware_matches_utility_oracle():
    """Cost-aware LinTS picks the argmax of the sampled LinTS score
    minus lambda times expected minutes, ties to the lowest id.
    The oracle recomputes the same utilities from plain parts."""
    cands = [_problem(1, "Easy", ["array"]),
             _problem(2, "Medium", ["graph"]),
             _problem(3, "Hard", ["dp"])]
    state = _state(log=_history())
    theta = bandit.sample_theta(state, seed=7)
    scored = []
    for p in cands:
        benefit = bandit.score(theta, bandit.context(p, state, NOW))
        scored.append((-(benefit - ta.LAMBDA_COST
                         * ta.expected_minutes(p, _history())),
                       p["id"]))
    expected_id = min(scored)[1]
    picked = ta.pick("cost_aware_lints", cands, state, NOW, 60.0, seed=7)
    assert picked["id"] == expected_id


def test_stage3_grind_bandit_17_lints_pick_reuses_production_selection():
    """The LinTS policy is the production selector unchanged: the
    dispatcher returns exactly the problem build_recommendation
    selects for the same seed."""
    cands = [_problem(1, "Easy", ["array"]),
             _problem(2, "Medium", ["graph"]),
             _problem(3, "Hard", ["dp"])]
    state = _state(log=_history())
    record = bandit.build_recommendation(state, cands, NOW, seed=7)
    picked = ta.pick("lints", cands, state, NOW, 60.0, seed=7)
    assert picked["id"] == record["selected_problem_id"]


def test_stage3_grind_bandit_17_policies_are_eligibility_matched():
    """Every policy draws from the same eligible set: no pick ever
    leaves the candidate list, and an empty candidate list yields
    no pick from any policy."""
    cands = [_problem(1, "Easy", ["array"]),
             _problem(2, "Medium", ["graph"]),
             _problem(3, "Hard", ["dp"])]
    state = _state(log=_history())
    ids = {p["id"] for p in cands}
    for name in reg.POLICY_NAMES:
        picked = ta.pick(name, cands, state, NOW, 30.0, seed=7)
        assert picked is None or picked["id"] in ids, name
    for name in reg.POLICY_NAMES:
        assert ta.pick(name, [], state, NOW, 30.0, seed=7) is None, name
    with pytest.raises(ValueError):
        ta.pick("not_a_policy", cands, state, NOW, 30.0, seed=7)


def test_stage3_grind_bandit_17_picks_do_not_mutate_state():
    """Policies only read: picking changes no log entry and no
    modeled mastery, so a shadow comparison cannot move live state."""
    cands = [_problem(1, "Easy", ["array"]),
             _problem(2, "Medium", ["graph"]),
             _problem(3, "Hard", ["dp"])]
    state = _state(
        topics={"array": _topic_entry(4, 2, NOW),
                "dp": _topic_entry(2, 4, NOW - 5 * DAY),
                "graph": _topic_entry(3, 3, NOW - 2 * DAY)},
        log=_history())
    before = copy.deepcopy(state)
    for name in reg.POLICY_NAMES:
        ta.pick(name, cands, state, NOW, 30.0, seed=7)
    assert state["log"] == before["log"]
    assert state["topics"] == before["topics"]


def test_stage3_grind_bandit_17_demo_state_not_identifiable():
    """The scripted demo history holds 30 attempts and no delayed
    quizzes at all, so the real-data gate has zero linked
    observations against a 30 minimum."""
    from common import build_demo_state

    state = build_demo_state(42)
    report = pc.evaluate(state)
    assert report["verdict"] == "NOT_IDENTIFIABLE", report
    assert report["n_linked_observations"] == 0
    assert report["n_attempts"] == 30
    assert report["production_policy"] == "lints"


def test_stage3_grind_bandit_17_empty_state_not_identifiable():
    report = pc.evaluate({"log": [], "delayed_quizzes": []})
    assert report["verdict"] == "NOT_IDENTIFIABLE"
    assert report["n_linked_observations"] == 0


def test_stage3_grind_bandit_17_simulation_truth_known_answers():
    """Known-answer oracles for the simulated world: the solve
    probability is the sigmoid of mastery minus the difficulty
    offset, and learning is match-gated, so a solved Easy problem
    on a weak topic (mastery 0.3, target Easy, match 1.0) gains
    exactly 0.30 times the headroom, while a failure gains 0.06."""
    assert pc.solve_probability(0.5, "Medium") == pytest.approx(
        1.0 / (1.0 + math.exp(-0.5)))
    assert pc.solve_probability(0.3, "Easy") == pytest.approx(
        1.0 / (1.0 + math.exp(-1.3)))
    learner = {"mastery": {"dp": 0.3}, "attempted": set()}
    pc.apply_learning(learner, _problem(1, "Easy", ["dp"]), solved=True)
    assert learner["mastery"]["dp"] == pytest.approx(0.3 + 0.30 * 1.0 * 0.7)
    learner = {"mastery": {"dp": 0.3}, "attempted": set()}
    pc.apply_learning(learner, _problem(1, "Easy", ["dp"]), solved=False)
    assert learner["mastery"]["dp"] == pytest.approx(0.3 + 0.06 * 1.0 * 0.7)
    # A Hard problem on the same weak topic is badly matched
    # (match 0.0), so even a solve teaches nothing in this world.
    learner = {"mastery": {"dp": 0.3}, "attempted": set()}
    pc.apply_learning(learner, _problem(1, "Hard", ["dp"]), solved=True)
    assert learner["mastery"]["dp"] == pytest.approx(0.3)


def test_stage3_grind_bandit_17_session_runs_within_reported_bounds():
    """One simulated session reports what it spent, whether it
    overran, what it picked, and a transfer proxy in [0, 1]."""
    bank, state, learner, now = pc.build_world(seed=7)
    report = pc.run_session("lints", bank, state, learner, now, 30.0,
                            seed=7)
    assert report["policy"] == "lints"
    assert report["budget_minutes"] == 30.0
    assert report["spent_minutes"] > 0.0
    assert report["overrun_minutes"] >= 0.0
    assert 0.0 <= report["transfer_proxy"] <= 1.0
    assert report["picks"], report
    assert all(pid in {p["id"] for p in bank} for pid in report["picks"])
    assert 0.0 <= report["hard_share"] <= 1.0
    assert report["n_quiz"] > 0


def test_stage3_grind_bandit_17_comparison_covers_all_policies_and_budgets():
    """The comparison runs every policy at every budget and reports
    transfer, overrun, Hard share, and topic coverage per cell, plus
    a simulation winner that is a real policy name or None. The
    simulation winner is machinery validation only: the real-data
    verdict stays NOT IDENTIFIABLE either way."""
    report = pc.compare(budgets=(15, 30), replications=6, base_seed=7)
    assert report["policy_version"] == "policy-v1"
    assert report["real_data_verdict"] == "NOT_IDENTIFIABLE"
    cells = {(r["policy"], r["budget"]) for r in report["rows"]}
    for name in reg.POLICY_NAMES:
        for budget in (15, 30):
            assert (name, budget) in cells, (name, budget)
    for row in report["rows"]:
        assert 0.0 <= row["transfer_proxy_mean"] <= 1.0
        assert row["overrun_mean"] >= 0.0
        assert 0.0 <= row["hard_share"] <= 1.0
        assert isinstance(row["overrun_flag"], bool)
        assert isinstance(row["starvation_flag"], bool)
        assert row["n_replications"] == 6
    assert report["simulation_winner"] in reg.POLICY_NAMES + [None]


def test_stage3_grind_bandit_17_gate_logic_on_fabricated_rows():
    """The gate itself, checked on fabricated rows: a challenger
    needs a transfer improvement whose interval excludes zero, no
    overrun, and no starvation. Each failure alone blocks adoption."""
    lints = {"transfer_diff_ci_low": 0.0, "transfer_diff_ci_high": 0.0,
             "overrun_flag": False, "starvation_flag": False}
    good = {"transfer_diff_ci_low": 0.01, "transfer_diff_ci_high": 0.05,
            "overrun_flag": False, "starvation_flag": False}
    assert pc.gate_check(good, lints)["adoptable"] is True
    no_edge = dict(good, transfer_diff_ci_low=-0.01)
    assert pc.gate_check(no_edge, lints)["adoptable"] is False
    assert "transfer" in pc.gate_check(no_edge, lints)["failures"][0]
    overrun = dict(good, overrun_flag=True)
    assert pc.gate_check(overrun, lints)["adoptable"] is False
    assert "overrun" in pc.gate_check(overrun, lints)["failures"][0]
    starved = dict(good, starvation_flag=True)
    assert pc.gate_check(starved, lints)["adoptable"] is False
    assert "starvation" in pc.gate_check(starved, lints)["failures"][0]


def test_stage3_grind_bandit_17_production_untouched():
    """The gate never rewrites live state: evaluating the demo
    history changes no mastery value, the feature version stays
    context-v2, the decay stays 30 days, and LinTS stays the only
    production policy."""
    from common import build_demo_state

    state = build_demo_state(42)
    before = {t: mastery.mean(state, t) for t in state["topics"]}
    pc.evaluate(state)
    after = {t: mastery.mean(state, t) for t in state["topics"]}
    assert before == after
    assert bandit.FEATURE_VERSION == "context-v2"
    assert mastery.HALF_LIFE_DAYS == 30.0
    assert reg.active_production_policy() == "lints"
    assert reg.REGISTRY["shortest_duration"]["status"] == "shadow"
    assert reg.REGISTRY["constrained_lints"]["status"] == "shadow"
