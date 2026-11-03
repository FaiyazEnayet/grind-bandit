"""Recommendation logging, reconstruction, and delayed transfer quizzes.

Every `next` pick is logged with its feature version, RNG seed, full
candidate set, posterior snapshot, and Monte Carlo selection probability
estimates, so a pick can be replayed from the record alone. Delayed
transfer quiz outcomes live in their own store and must never feed
mastery, the bandit posterior, or the model-derived reward.
"""

import copy
import json

import pytest

from grindbandit import bandit, cli, problems, state as st

DAY = 86400.0


@pytest.fixture
def iso_state(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    monkeypatch.setenv("GRIND_BANDIT_STATE", str(p))
    return p


def _only_record():
    s = st.load()
    assert len(s["recommendations"]) == 1
    return s["recommendations"][0]


def test_next_logs_a_complete_recommendation_record(iso_state):
    cli.main(["next", "--seed", "11"])
    rec = _only_record()
    assert rec["id"]
    assert rec["feature_version"] == bandit.FEATURE_VERSION
    assert rec["feature_names"] == bandit.FEATURES
    assert rec["seed"] == 11
    assert rec["selected_problem_id"] in {c["problem_id"] for c in rec["candidates"]}
    selected = [c for c in rec["candidates"] if c["selected"]]
    assert len(selected) == 1
    assert selected[0]["problem_id"] == rec["selected_problem_id"]
    est = rec["probability_estimate"]
    assert est["samples"] == bandit.PROBABILITY_SAMPLES
    assert est["seed"] != rec["seed"]  # estimation uses its own RNG stream
    for c in rec["candidates"]:
        assert len(c["context"]) == bandit.D
        p = c["estimated_selection_probability"]
        assert 0.0 < p <= 1.0
    total = sum(c["estimated_selection_probability"] for c in rec["candidates"])
    assert abs(total - 1.0) < 1e-9


def test_recommendation_replays_to_the_same_pick_from_the_log_alone(iso_state):
    # Give the posterior some history first so the draw is not the prior.
    cli.main(["log", "1", "solved"])
    cli.main(["log", "322", "failed"])
    cli.main(["next", "--seed", "5"])
    rec = _only_record()
    round_tripped = json.loads(json.dumps(rec))  # exactly what was persisted
    assert bandit.replay_selection(round_tripped) == rec["selected_problem_id"]


def test_replay_rejects_a_record_from_another_feature_version(iso_state):
    cli.main(["next", "--seed", "5"])
    rec = copy.deepcopy(_only_record())
    rec["feature_version"] = "context-v999"
    with pytest.raises(ValueError):
        bandit.replay_selection(rec)


def test_next_without_a_seed_still_logs_the_seed_it_used(iso_state):
    cli.main(["next"])
    rec = _only_record()
    assert isinstance(rec["seed"], int)
    assert bandit.replay_selection(rec) == rec["selected_problem_id"]


def test_log_links_attempt_to_recommendation_and_schedules_quiz(iso_state, capsys):
    cli.main(["next", "--seed", "3"])
    rec = _only_record()
    pid = rec["selected_problem_id"]
    capsys.readouterr()
    cli.main(["log", str(pid), "solved", "--minutes", "25"])
    out = capsys.readouterr().out
    assert "model-derived reward" in out
    assert "delayed transfer quiz scheduled" in out
    s = st.load()
    entry = s["log"][-1]
    assert entry["recommendation_id"] == rec["id"]
    assert entry["reward_kind"] == "model_derived_mastery_change"
    assert len(s["delayed_quizzes"]) == 1
    q = s["delayed_quizzes"][0]
    assert q["status"] == "pending" and q["outcome"] is None
    assert q["source_problem_id"] == pid
    assert q["source_recommendation_id"] == rec["id"]
    assert q["quiz_problem_id"] != pid
    assert q["due_ts"] - q["scheduled_ts"] == pytest.approx(3 * DAY)
    quiz_problem = problems.by_id(problems.load(), q["quiz_problem_id"])
    assert q["topic"] in quiz_problem["topics"]


def test_delayed_outcome_is_separate_and_never_feeds_the_model(iso_state):
    cli.main(["log", "1", "solved"])
    s = st.load()
    q = s["delayed_quizzes"][0]
    topics_before = copy.deepcopy(s["topics"])
    bandit_before = copy.deepcopy(s["bandit"])
    log_before = copy.deepcopy(s["log"])
    # Recorded immediately, so the explicit early override is required;
    # the separation guarantee being tested is unchanged by the timing.
    cli.main(["quiz", q["id"], "solved", "--early"])
    s2 = st.load()
    q2 = s2["delayed_quizzes"][0]
    assert q2["status"] == "completed"
    assert q2["outcome"]["solved"] is True
    assert "recorded_ts" in q2["outcome"]
    assert s2["topics"] == topics_before
    assert s2["bandit"] == bandit_before
    assert s2["log"] == log_before


def test_quiz_outcome_cannot_be_recorded_twice(iso_state):
    cli.main(["log", "1", "solved"])
    qid = st.load()["delayed_quizzes"][0]["id"]
    cli.main(["quiz", qid, "failed", "--early"])
    with pytest.raises(SystemExit):
        cli.main(["quiz", qid, "solved"])
    q = st.load()["delayed_quizzes"][0]
    assert q["outcome"]["solved"] is False


def test_quiz_unknown_id_fails(iso_state):
    with pytest.raises(SystemExit):
        cli.main(["quiz", "quiz-nope", "solved"])


def test_stats_lists_pending_delayed_quizzes(iso_state, capsys):
    cli.main(["log", "1", "solved"])
    capsys.readouterr()
    cli.main(["stats"])
    out = capsys.readouterr().out
    assert "Delayed transfer quizzes pending" in out
    assert "observational" in out


def test_state_files_without_the_new_fields_still_work(iso_state, capsys):
    legacy = {"topics": {}, "problems": {}, "log": [], "bandit": {}}
    iso_state.write_text(json.dumps(legacy))
    cli.main(["next", "--seed", "2"])
    out = capsys.readouterr().out
    assert "Why this one:" in out
    cli.main(["log", "322", "failed"])
    s = st.load()
    assert len(s["recommendations"]) == 1
    assert len(s["delayed_quizzes"]) == 1
    # An attempt on a problem no recommendation selected stays unlinked.
    assert "recommendation_id" not in s["log"][-1]


# --- Stage 3 GB10: delayed assessment timing, novelty and adherence ---
#
# A delayed quiz measures retention after the delay, so an outcome
# recorded before the due date is not the measurement the schedule
# promised. Recording is rejected by default until the quiz comes due;
# an explicit override exists for the cases where the user did the quiz
# early on purpose. Each recorded outcome carries the actual delay and
# an early/on-time/late status. Quiz problems are drawn from unused
# items first: a problem already assigned to an earlier quiz is not
# assigned again while a never-attempted, never-assigned alternative
# exists, and reuse after exhaustion is stated on the record.

T0 = 1_700_000_000.0


def _mini_bank():
    return [
        {"id": 1, "title": "One", "difficulty": "Easy", "topics": ["array"]},
        {"id": 2, "title": "Two", "difficulty": "Easy", "topics": ["array"]},
        {"id": 3, "title": "Three", "difficulty": "Medium", "topics": ["array"]},
        {"id": 4, "title": "Four", "difficulty": "Medium", "topics": ["array"]},
        {"id": 9, "title": "Nine", "difficulty": "Hard", "topics": ["graph"]},
    ]


def _blank_state():
    return {"topics": {}, "problems": {}, "log": [], "bandit": {},
            "recommendations": [], "delayed_quizzes": []}


def test_stage3_grind_bandit_10_early_outcome_rejected_by_default():
    s = _blank_state()
    bank = _mini_bank()
    q = st.schedule_delayed_quiz(s, bank, bank[0], "array", T0)
    assert q is not None
    with pytest.raises(ValueError):
        st.record_delayed_outcome(s, q["id"], True, T0 + 3600.0)
    after = st.find_delayed_quiz(s, q["id"])
    assert after["status"] == "pending"
    assert after["outcome"] is None


def test_stage3_grind_bandit_10_cli_rejects_early_outcome_without_override(iso_state):
    cli.main(["log", "1", "solved"])
    qid = st.load()["delayed_quizzes"][0]["id"]
    with pytest.raises(SystemExit):
        cli.main(["quiz", qid, "solved"])
    q = st.load()["delayed_quizzes"][0]
    assert q["status"] == "pending"
    assert q["outcome"] is None


def test_stage3_grind_bandit_10_due_and_late_outcomes_record_timing():
    s = _blank_state()
    bank = _mini_bank()
    q1 = st.schedule_delayed_quiz(s, bank, bank[0], "array", T0)
    done = st.record_delayed_outcome(s, q1["id"], True, q1["due_ts"])
    assert done["status"] == "completed"
    oc = done["outcome"]
    assert oc["solved"] is True
    assert oc["recorded_ts"] == q1["due_ts"]
    assert oc["delay_seconds"] == pytest.approx(3 * DAY)
    assert oc["timing"] == "on_time"
    q2 = st.schedule_delayed_quiz(s, bank, bank[1], "array", T0)
    late = st.record_delayed_outcome(s, q2["id"], False, q2["due_ts"] + 2 * DAY)
    assert late["outcome"]["timing"] == "late"
    assert late["outcome"]["delay_seconds"] == pytest.approx(5 * DAY)


def test_stage3_grind_bandit_10_early_override_records_and_marks_early(iso_state):
    s = _blank_state()
    bank = _mini_bank()
    q = st.schedule_delayed_quiz(s, bank, bank[0], "array", T0)
    done = st.record_delayed_outcome(s, q["id"], False, T0 + DAY, allow_early=True)
    assert done["status"] == "completed"
    assert done["outcome"]["timing"] == "early"
    assert done["outcome"]["delay_seconds"] == pytest.approx(DAY)
    # The CLI exposes the same override as an explicit flag.
    cli.main(["log", "1", "solved"])
    qid = st.load()["delayed_quizzes"][0]["id"]
    cli.main(["quiz", qid, "solved", "--early"])
    q2 = st.load()["delayed_quizzes"][0]
    assert q2["status"] == "completed"
    assert q2["outcome"]["timing"] == "early"


def test_stage3_grind_bandit_10_quiz_outcome_mutates_model_state_by_zero(iso_state):
    cli.main(["log", "1", "solved"])
    cli.main(["log", "322", "failed"])
    s = st.load()
    snapshot = {k: copy.deepcopy(s[k])
                for k in ("topics", "bandit", "log", "problems", "recommendations")}
    q = s["delayed_quizzes"][0]
    st.record_delayed_outcome(s, q["id"], True, q["due_ts"])
    for k, before in snapshot.items():
        assert s[k] == before, k


def test_stage3_grind_bandit_10_unused_problems_prioritized_over_assigned():
    s = _blank_state()
    bank = _mini_bank()
    picked = []
    for _ in range(3):
        q = st.schedule_delayed_quiz(s, bank, bank[0], "array", T0)
        picked.append(q["quiz_problem_id"])
    # Three unused alternatives exist, so no problem is assigned twice.
    assert picked == [2, 3, 4]


def test_stage3_grind_bandit_10_attempted_problems_are_last_resort():
    s = _blank_state()
    s["problems"] = {"3": {"attempts": 1, "solved": 0, "last_ts": T0,
                            "ever_solved": False}}
    bank = _mini_bank()
    q1 = st.schedule_delayed_quiz(s, bank, bank[0], "array", T0)
    q2 = st.schedule_delayed_quiz(s, bank, bank[0], "array", T0)
    # 2 is unused; then 4 (unused) beats both 2 (already assigned) and
    # 3 (already attempted).
    assert q1["quiz_problem_id"] == 2
    assert q2["quiz_problem_id"] == 4


def test_stage3_grind_bandit_10_exhaustion_is_stated_explicitly():
    bank = _mini_bank()[:2]  # one source problem, one alternative
    s = _blank_state()
    q1 = st.schedule_delayed_quiz(s, bank, bank[0], "array", T0)
    assert q1["quiz_problem_id"] == 2
    assert q1["assignment"]["novel"] is True
    assert q1["assignment"]["pool_exhausted"] is False
    assert q1["assignment"]["prior_assignments"] == 0
    q2 = st.schedule_delayed_quiz(s, bank, bank[0], "array", T0)
    assert q2["quiz_problem_id"] == 2  # the only alternative, reused
    assert q2["assignment"]["novel"] is False
    assert q2["assignment"]["pool_exhausted"] is True
    assert q2["assignment"]["prior_assignments"] == 1


def test_stage3_grind_bandit_10_old_format_quiz_records_still_work():
    s = _blank_state()
    old_pending = {
        "id": "quiz-old1", "kind": "delayed_transfer_quiz",
        "scheduled_ts": T0, "due_ts": T0 + 3 * DAY, "delay_days": 3,
        "topic": "array", "source_problem_id": 1,
        "source_recommendation_id": None, "quiz_problem_id": 2,
        "quiz_title": "Two", "status": "pending", "outcome": None,
    }
    old_done = dict(old_pending, id="quiz-old2", status="completed",
                    outcome={"solved": False, "recorded_ts": T0 + 4 * DAY})
    s["delayed_quizzes"] = [old_pending, old_done]
    assert [q["id"] for q in st.pending_delayed_quizzes(s)] == ["quiz-old1"]
    done = st.record_delayed_outcome(s, "quiz-old1", True, T0 + 3 * DAY)
    assert done["status"] == "completed"
    assert done["outcome"]["solved"] is True
    # An old completed record reads back as-is and cannot be re-recorded.
    assert st.find_delayed_quiz(s, "quiz-old2")["outcome"]["solved"] is False
    assert st.record_delayed_outcome(s, "quiz-old2", True, T0 + 5 * DAY) is None


def test_stage3_grind_bandit_10_stats_reports_recorded_outcome_timing(iso_state, capsys):
    cli.main(["log", "1", "solved"])
    qid = st.load()["delayed_quizzes"][0]["id"]
    cli.main(["quiz", qid, "solved", "--early"])
    capsys.readouterr()
    cli.main(["stats"])
    out = capsys.readouterr().out
    assert "Delayed transfer quiz outcomes" in out
    assert "early" in out
