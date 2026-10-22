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
    cli.main(["quiz", q["id"], "solved"])
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
    cli.main(["quiz", qid, "failed"])
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
