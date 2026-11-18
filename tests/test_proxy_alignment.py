"""Stage 3 GB13: proxy reward alignment laboratory.

The bandit optimizes a model-derived reward (change in modeled mastery
of the weakest topic). That reward can rise mechanically even when
delayed transfer on a different problem days later does not move. These
tests pin the offline diagnostics that check whether ranking by the
reward predicts delayed transfer outcomes.

The fixtures are hand-built and known-answer: exact Spearman and AUC
values computed by hand, a small joined state where every delay,
novelty flag, exposure count, and missing outcome is asserted, and a
controlled simulation with a known ground truth (aligned vs misaligned
worlds) used only to validate the diagnostics. Simulated rows are never
presented as real practice data, and the real-data gate is pinned:
with no linked observations the verdict is NOT IDENTIFIABLE.
"""

import copy
import math

import pytest

from grindbandit.research import proxy_alignment as pa

DAY = 86400.0
T0 = 1_700_000_000.0


def _problem(pid=1, topic="array"):
    return {"id": pid, "title": f"P{pid}", "difficulty": "Easy",
            "topics": [topic]}


def _blank_state():
    return {"schema_version": 1, "topics": {}, "problems": {}, "log": [],
            "bandit": {}, "recommendations": [], "delayed_quizzes": []}


def _attempt(aid, pid, topic, solved, reward, ts):
    return {"attempt_id": aid, "ts": ts, "problem_id": pid, "title": f"P{pid}",
            "difficulty": "Easy", "topics": [topic], "solved": solved,
            "minutes": 20, "reward": reward,
            "reward_kind": "model_derived_mastery_change"}


def _quiz(qid, source_aid, quiz_pid, topic, scheduled_ts, novel,
          prior_assignments, outcome=None, delay_seconds=None,
          timing=None):
    q = {"id": qid, "kind": "delayed_transfer_quiz",
         "scheduled_ts": scheduled_ts,
         "due_ts": scheduled_ts + 3 * DAY, "delay_days": 3, "topic": topic,
         "source_problem_id": 1, "source_attempt_id": source_aid,
         "source_recommendation_id": None, "quiz_problem_id": quiz_pid,
         "quiz_title": f"P{quiz_pid}",
         "assignment": {"novel": novel, "prior_assignments": prior_assignments,
                        "pool_exhausted": False},
         "status": "pending" if outcome is None else "completed",
         "outcome": outcome}
    if outcome is not None:
        q["outcome"] = {"solved": outcome, "recorded_ts": scheduled_ts + delay_seconds,
                        "delay_seconds": delay_seconds, "timing": timing}
    return q


def _joined_state():
    """Three attempts, two quizzes, one legacy quiz, one legacy attempt.

    att-1 solved with reward +0.40, quiz on novel problem 7, completed
    solved after exactly 3 days. att-2 failed with reward -0.25, quiz on
    reused problem 3 (attempted once before the quiz), still pending.
    att-3 solved with reward +0.10 and no quiz at all. One legacy quiz
    names no source attempt, and one legacy attempt carries no id.
    """
    state = _blank_state()
    state["log"] = [
        _attempt("att-1", 1, "array", True, 0.40, T0),
        _attempt("att-2", 2, "dp", False, -0.25, T0 + DAY),
        _attempt("att-3", 5, "array", True, 0.10, T0 + 2 * DAY),
        {"ts": T0 + 3 * DAY, "problem_id": 3, "title": "P3",
         "difficulty": "Easy", "topics": ["dp"], "solved": True,
         "minutes": 20, "reward": 0.20,
         "reward_kind": "model_derived_mastery_change"},
    ]
    # One prior attempt on problem 3, before att-2's quiz was scheduled.
    state["log"].append(
        _attempt("att-0", 3, "dp", True, 0.15, T0 - DAY))
    state["delayed_quizzes"] = [
        _quiz("quiz-1", "att-1", 7, "array", T0, True, 0,
              outcome=True, delay_seconds=3 * DAY, timing="on_time"),
        _quiz("quiz-2", "att-2", 3, "dp", T0 + DAY, False, 1),
        _quiz("quiz-legacy", None, 9, "array", T0, True, 0),
    ]
    return state


def _linked_rows(n, solved_every=2):
    """n completed linked rows with steadily rising rewards."""
    rows = []
    for i in range(n):
        rows.append({
            "attempt_id": f"att-{i}", "attempt_ts": T0 + i * DAY,
            "reward": -0.5 + i / max(n - 1, 1),
            "source_solved": i % 2 == 0, "topic": "array",
            "delay_days": 3.0, "timing": "on_time", "novel": True,
            "prior_assignments": 0, "pool_exhausted": False,
            "quiz_problem_id": 100 + i, "quiz_problem_prior_attempts": 0,
            "topic_prior_attempts": i, "outcome_solved": (i % solved_every == 0),
            "outcome_status": "completed",
        })
    return rows


# --- Known-answer statistics -------------------------------------------


def test_stage3_grind_bandit_13_spearman_perfect_positive_and_negative():
    assert pa.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert pa.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_stage3_grind_bandit_13_spearman_tied_ranks_known_answer():
    # Ranks of x are [1.5, 1.5, 3], ranks of y are [1, 2, 3];
    # Pearson of the ranks is 0.8660254037844387 by hand.
    assert pa.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254037844387)


def test_stage3_grind_bandit_13_spearman_undefined_for_constant_input():
    assert pa.spearman([1, 1, 1], [1, 2, 3]) is None
    assert pa.spearman([1, 2, 3], [2, 2, 2]) is None
    assert pa.spearman([1], [1]) is None


def test_stage3_grind_bandit_13_auc_known_answer():
    # Positive scores 0.35 and 0.8 against negatives 0.1 and 0.4:
    # 3 of 4 pairs rank the positive higher, so AUC is 0.75.
    assert pa.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert pa.auc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert pa.auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_stage3_grind_bandit_13_auc_undefined_for_single_class():
    assert pa.auc([0.1, 0.2], [1, 1]) is None
    assert pa.auc([0.1, 0.2], [0, 0]) is None


# --- Join: delay, novelty, exposure, missing outcomes -------------------


def test_stage3_grind_bandit_13_join_preserves_delay_novelty_and_exposure():
    joined = pa.link_attempts_to_quizzes(_joined_state())
    assert joined["n_attempts"] == 5
    assert joined["n_attempts_with_quiz"] == 2
    assert joined["n_attempts_without_quiz"] == 3
    assert joined["n_quizzes"] == 3
    assert joined["n_completed"] == 1
    assert joined["n_pending"] == 1  # quiz-2; the legacy quiz is attrition, not a row
    rows = {r["attempt_id"]: r for r in joined["rows"]}
    first = rows["att-1"]
    assert first["delay_days"] == pytest.approx(3.0)
    assert first["timing"] == "on_time"
    assert first["novel"] is True
    assert first["prior_assignments"] == 0
    assert first["quiz_problem_prior_attempts"] == 0
    assert first["outcome_solved"] is True
    second = rows["att-2"]
    assert second["novel"] is False
    assert second["prior_assignments"] == 1
    assert second["quiz_problem_prior_attempts"] == 1
    assert second["outcome_solved"] is None
    assert second["outcome_status"] == "pending"


def test_stage3_grind_bandit_13_join_counts_legacy_records_as_attrition():
    joined = pa.link_attempts_to_quizzes(_joined_state())
    assert joined["n_attempts_without_id"] == 1
    assert joined["n_quizzes_without_source_attempt"] == 1
    assert joined["n_linked"] == 2
    assert joined["n_linked_completed"] == 1


def test_stage3_grind_bandit_13_join_ignores_quiz_naming_unknown_attempt():
    state = _blank_state()
    state["delayed_quizzes"] = [
        _quiz("quiz-x", "att-missing", 9, "array", T0, True, 0,
              outcome=True, delay_seconds=3 * DAY, timing="on_time"),
    ]
    joined = pa.link_attempts_to_quizzes(state)
    assert joined["n_linked"] == 0
    assert joined["n_quizzes_without_source_attempt"] == 1


def test_stage3_grind_bandit_13_join_tolerates_state_without_quiz_keys():
    state = {"topics": {}, "problems": {}, "log": [], "bandit": {}}
    joined = pa.link_attempts_to_quizzes(state)
    assert joined["n_linked"] == 0
    assert joined["n_attempts"] == 0


# --- Gate: NOT IDENTIFIABLE without enough real linked observations -----


def test_stage3_grind_bandit_13_empty_state_is_not_identifiable():
    report = pa.analyze(_blank_state())
    assert report["verdict"] == "NOT IDENTIFIABLE"
    assert report["n_linked_completed"] == 0
    assert report["diagnostics"] is None


def test_stage3_grind_bandit_13_small_linked_sample_is_not_identifiable():
    report = pa.diagnose(_linked_rows(12))
    assert report["verdict"] == "NOT IDENTIFIABLE"
    assert report["n_linked_completed"] == 12
    assert "30" in report["reason"]


def test_stage3_grind_bandit_13_demo_history_has_no_linked_observations():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from common import build_demo_state

    state = build_demo_state(seed=42)
    assert len(state["log"]) == 30
    report = pa.analyze(state)
    assert report["verdict"] == "NOT IDENTIFIABLE"
    assert report["n_linked_completed"] == 0
    assert report["attrition"]["n_quizzes"] == 0


# --- Holdout: predeclared, time-ordered, never used for tuning ----------


def test_stage3_grind_bandit_13_holdout_is_latest_quarter_by_time():
    rows = _linked_rows(40)
    train, holdout = pa.split_holdout(rows, holdout_fraction=0.25)
    assert len(train) == 30
    assert len(holdout) == 10
    assert max(r["attempt_ts"] for r in train) < min(r["attempt_ts"] for r in holdout)
    assert [r["attempt_id"] for r in train + holdout] == [r["attempt_id"] for r in rows]


def test_stage3_grind_bandit_13_holdout_below_minimum_is_not_identifiable():
    rows = _linked_rows(32)
    train, holdout = pa.split_holdout(rows, holdout_fraction=0.25)
    assert len(holdout) == 8
    report = pa.diagnose(rows)
    assert report["verdict"] == "NOT IDENTIFIABLE"


# --- Sensitivity: missing outcomes are bounds, never failures ------------


def test_stage3_grind_bandit_13_missing_outcomes_bound_not_failed():
    rows = _linked_rows(40)
    for r in rows[20:]:
        r["outcome_solved"] = None
        r["outcome_status"] = "pending"
    bounds = pa.sensitivity_bounds(rows)
    assert bounds["n_completed"] == 20
    assert bounds["n_missing"] == 20
    assert bounds["observed_rate"] == pytest.approx(0.5)
    assert bounds["worst_case_rate"] == pytest.approx(0.25)
    assert bounds["best_case_rate"] == pytest.approx(0.75)
    assert bounds["worst_case_rate"] <= bounds["observed_rate"] <= bounds["best_case_rate"]


def test_stage3_grind_bandit_13_sensitivity_bounds_cover_spearman():
    rows = _linked_rows(40)
    for r in rows[20:]:
        r["outcome_solved"] = None
        r["outcome_status"] = "pending"
    bounds = pa.sensitivity_bounds(rows)
    assert bounds["spearman_worst"] <= bounds["spearman_observed"] <= bounds["spearman_best"]


# --- Stratified failure maps and calibration ----------------------------


def test_stage3_grind_bandit_13_stratified_map_groups_by_novelty_and_timing():
    rows = _linked_rows(40)
    for r in rows[:20]:
        r["novel"] = False
        r["timing"] = "late"
    strata = pa.stratified_failure_map(rows, ["novel", "timing"])
    keyed = {(s["novel"], s["timing"]): s for s in strata}
    assert keyed[(False, "late")]["n"] == 20
    assert keyed[(True, "on_time")]["n"] == 20
    assert keyed[(False, "late")]["delayed_solve_rate"] == pytest.approx(0.5)


def test_stage3_grind_bandit_13_calibration_terciles_cover_all_completed():
    rows = _linked_rows(40)
    terciles = pa.calibration_by_tercile(rows)
    assert len(terciles) == 3
    assert sum(t["n"] for t in terciles) == 40
    assert terciles[0]["mean_reward"] < terciles[1]["mean_reward"] < terciles[2]["mean_reward"]


# --- Controlled simulation: known ground truth validates diagnostics -----


def test_stage3_grind_bandit_13_aligned_world_shows_positive_association():
    rows = pa.simulate_linked_rows(400, alignment="aligned", seed=7)
    report = pa.diagnose(rows)
    assert report["verdict"] == "IDENTIFIABLE"
    assert report["diagnostics"]["spearman"] > 0.3
    assert report["diagnostics"]["auc"] > 0.6
    assert report["diagnostics"]["holdout"]["spearman"] > 0.2


def test_stage3_grind_bandit_13_misaligned_world_shows_no_association():
    rows = pa.simulate_linked_rows(400, alignment="misaligned", seed=7)
    report = pa.diagnose(rows)
    assert report["verdict"] == "IDENTIFIABLE"
    assert abs(report["diagnostics"]["spearman"]) < 0.2
    assert abs(report["diagnostics"]["auc"] - 0.5) < 0.1


def test_stage3_grind_bandit_13_simulation_is_reproducible_and_labeled():
    first = pa.simulate_linked_rows(50, alignment="aligned", seed=11)
    second = pa.simulate_linked_rows(50, alignment="aligned", seed=11)
    assert first == second
    assert all(r["outcome_status"] == "completed" for r in first)
    assert all(r.get("simulated") is True for r in first)


def test_stage3_grind_bandit_13_simulation_rejects_unknown_alignment():
    with pytest.raises(ValueError):
        pa.simulate_linked_rows(10, alignment="sideways", seed=1)


# --- Read-only analysis ---------------------------------------------------


def test_stage3_grind_bandit_13_analyze_never_mutates_state():
    state = _joined_state()
    before = copy.deepcopy(state)
    pa.analyze(state)
    assert state == before
