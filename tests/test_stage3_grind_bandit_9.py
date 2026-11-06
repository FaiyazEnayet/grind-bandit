"""Stage 3 GB9: precise attempt/recommendation lineage.

Attempt entries used to carry no id of their own, and the link back to a
recommendation was reconstructed after the fact from the problem id
alone: the most recent recommendation that had ever selected the
problem was written onto the attempt (and its delayed quiz) as if
confirmed, even for a problem the user chose themselves, an old pick,
or a problem recommended more than once. These tests pin the precise
lineage: every attempt has an immutable attempt_id, an explicit
--recommendation-id is validated before anything is logged, auto-linking
only infers from a recent, globally latest, unconsumed recommendation
and says so on the record, and linkage metadata never feeds the model.
"""

import copy
import json

import numpy as np
import pytest

from grindbandit import bandit, cli, state as st

DAY = 86400.0
T0 = 1_700_000_000.0


@pytest.fixture
def iso_state(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    monkeypatch.setenv("GRIND_BANDIT_STATE", str(p))
    return p


def _blank_state():
    return {"schema_version": 1, "topics": {}, "problems": {}, "log": [],
            "bandit": {}, "recommendations": [], "delayed_quizzes": []}


def _problem(pid=1, topic="array"):
    return {"id": pid, "title": f"P{pid}", "difficulty": "Easy",
            "topics": [topic]}


def _rec(rid, pid, ts):
    return {"id": rid, "ts": ts, "selected_problem_id": pid}


# --- attempt ids and quiz source lineage ---------------------------------


def test_stage3_grind_bandit_9_every_attempt_gets_a_unique_attempt_id():
    s = _blank_state()
    e1 = st.record_attempt(s, _problem(1), True, 20, 0.1, T0)
    e2 = st.record_attempt(s, _problem(1), False, 35, -0.2, T0 + 60.0)
    assert e1["attempt_id"] and e2["attempt_id"]
    assert e1["attempt_id"] != e2["attempt_id"]
    assert [e["attempt_id"] for e in s["log"]] == [
        e1["attempt_id"], e2["attempt_id"]]


def test_stage3_grind_bandit_9_two_attempts_same_problem_never_alias_source_attempt_id():
    s = _blank_state()
    bank = [_problem(1), _problem(2), _problem(3)]
    e1 = st.record_attempt(s, bank[0], True, 20, 0.1, T0)
    q1 = st.schedule_delayed_quiz(
        s, bank, bank[0], "array", T0, source_attempt_id=e1["attempt_id"])
    e2 = st.record_attempt(s, bank[0], True, 20, 0.1, T0 + DAY)
    q2 = st.schedule_delayed_quiz(
        s, bank, bank[0], "array", T0 + DAY,
        source_attempt_id=e2["attempt_id"])
    assert q1 is not None and q2 is not None
    assert q1["source_attempt_id"] == e1["attempt_id"]
    assert q2["source_attempt_id"] == e2["attempt_id"]
    assert q1["source_attempt_id"] != q2["source_attempt_id"]


# --- explicit recommendation ids are validated ----------------------------


def test_stage3_grind_bandit_9_explicit_link_is_recorded_as_explicit():
    s = _blank_state()
    s["recommendations"].append(_rec("rec-a", 1, T0))
    rec = st.validate_recommendation_link(s, "rec-a", 1)
    assert rec["id"] == "rec-a"
    e = st.record_attempt(s, _problem(1), True, 20, 0.1, T0 + 10.0,
                          recommendation_id="rec-a",
                          linkage_method="explicit",
                          recommendation_status={"exposed": True,
                                                 "accepted": True})
    assert e["recommendation_id"] == "rec-a"
    assert e["linkage_method"] == "explicit"
    assert e["recommendation_status"] == {"exposed": True, "accepted": True}


def test_stage3_grind_bandit_9_mismatched_recommendation_id_rejected():
    s = _blank_state()
    s["recommendations"].append(_rec("rec-a", 2, T0))  # selected problem 2
    with pytest.raises(ValueError):
        st.validate_recommendation_link(s, "rec-a", 1)  # logging problem 1


def test_stage3_grind_bandit_9_unknown_recommendation_id_rejected():
    s = _blank_state()
    with pytest.raises(ValueError):
        st.validate_recommendation_link(s, "rec-nope", 1)


def test_stage3_grind_bandit_9_stale_recommendation_id_rejected():
    s = _blank_state()
    s["recommendations"].append(_rec("rec-old", 1, T0))
    s["recommendations"].append(_rec("rec-new", 1, T0 + 60.0))
    with pytest.raises(ValueError):
        st.validate_recommendation_link(s, "rec-old", 1)
    # The latest one still validates.
    assert st.validate_recommendation_link(s, "rec-new", 1)["id"] == "rec-new"


def test_stage3_grind_bandit_9_consumed_recommendation_id_rejected():
    s = _blank_state()
    s["recommendations"].append(_rec("rec-a", 1, T0))
    st.record_attempt(s, _problem(1), True, 20, 0.1, T0 + 10.0,
                      recommendation_id="rec-a", linkage_method="explicit")
    with pytest.raises(ValueError):
        st.validate_recommendation_link(s, "rec-a", 1)


def test_stage3_grind_bandit_9_cli_rejects_mismatched_id_and_logs_nothing(
        iso_state):
    cli.main(["next", "--seed", "3"])
    s = st.load()
    rec = s["recommendations"][0]
    other = 322 if rec["selected_problem_id"] != 322 else 1
    with pytest.raises(SystemExit):
        cli.main(["log", str(other), "solved",
                  "--recommendation-id", rec["id"]])
    s2 = st.load()
    assert s2["log"] == []
    assert s2["problems"] == {}
    assert s2["bandit"] == s["bandit"]


def test_stage3_grind_bandit_9_cli_explicit_recommendation_id_links(
        iso_state):
    cli.main(["next", "--seed", "3"])
    rec = st.load()["recommendations"][0]
    cli.main(["log", str(rec["selected_problem_id"]), "solved",
              "--recommendation-id", rec["id"]])
    s = st.load()
    entry = s["log"][-1]
    assert entry["recommendation_id"] == rec["id"]
    assert entry["linkage_method"] == "explicit"
    assert entry["attempt_id"]
    q = s["delayed_quizzes"][-1]
    assert q["source_attempt_id"] == entry["attempt_id"]
    assert q["source_recommendation_id"] == rec["id"]


# --- inferred auto-linking is narrow and labeled ---------------------------


def test_stage3_grind_bandit_9_inference_links_recent_latest_unconsumed():
    s = _blank_state()
    s["recommendations"].append(_rec("rec-a", 1, T0))
    rec = st.infer_recommendation_link(s, 1, T0 + 3600.0)
    assert rec is not None and rec["id"] == "rec-a"


def test_stage3_grind_bandit_9_inference_ignores_old_recommendation():
    s = _blank_state()
    s["recommendations"].append(_rec("rec-a", 1, T0))
    # Two days later the pick is no longer recent: no inference.
    assert st.infer_recommendation_link(s, 1, T0 + 2 * DAY) is None


def test_stage3_grind_bandit_9_inference_requires_globally_latest():
    s = _blank_state()
    s["recommendations"].append(_rec("rec-a", 1, T0))
    s["recommendations"].append(_rec("rec-b", 2, T0 + 60.0))
    # rec-a matches the problem but a newer recommendation exists.
    assert st.infer_recommendation_link(s, 1, T0 + 120.0) is None


def test_stage3_grind_bandit_9_inference_skips_consumed_recommendation():
    s = _blank_state()
    s["recommendations"].append(_rec("rec-a", 1, T0))
    st.record_attempt(s, _problem(1), True, 20, 0.1, T0 + 10.0,
                      recommendation_id="rec-a", linkage_method="explicit")
    assert st.infer_recommendation_link(s, 1, T0 + 20.0) is None


def test_stage3_grind_bandit_9_manually_chosen_attempt_remains_unlinked(
        iso_state):
    # No recommendation was ever made, so nothing can be linked.
    cli.main(["log", "1", "solved"])
    s = st.load()
    entry = s["log"][-1]
    assert "recommendation_id" not in entry
    assert "linkage_method" not in entry
    assert entry["attempt_id"]
    q = s["delayed_quizzes"][-1]
    assert q["source_recommendation_id"] is None
    assert q["source_attempt_id"] == entry["attempt_id"]


def test_stage3_grind_bandit_9_cli_auto_link_is_labeled_inferred(iso_state):
    cli.main(["next", "--seed", "3"])
    rec = st.load()["recommendations"][0]
    cli.main(["log", str(rec["selected_problem_id"]), "solved"])
    entry = st.load()["log"][-1]
    assert entry["recommendation_id"] == rec["id"]
    assert entry["linkage_method"] == "inferred"


# --- historical entries and model separation -------------------------------


def test_stage3_grind_bandit_9_old_entries_still_load(iso_state):
    legacy = {
        "topics": {}, "problems": {},
        "log": [{"ts": T0, "problem_id": 1, "title": "Old", "solved": True,
                 "minutes": 10, "reward": 0.5}],
        "bandit": {},
        "recommendations": [{"id": "rec-legacy", "ts": T0,
                             "selected_problem_id": 1}],
        "delayed_quizzes": [{"id": "quiz-legacy", "status": "pending",
                             "source_problem_id": 1,
                             "source_recommendation_id": "rec-legacy",
                             "quiz_problem_id": 2, "outcome": None}],
    }
    iso_state.write_text(json.dumps(legacy))
    s = st.load()
    # Nothing is fabricated for the historical attempt or quiz.
    assert "attempt_id" not in s["log"][0]
    assert "linkage_method" not in s["log"][0]
    assert "source_attempt_id" not in s["delayed_quizzes"][0]
    # And the state still accepts new, fully linked attempts.
    cli.main(["log", "1", "solved"])
    s2 = st.load()
    assert s2["log"][-1]["attempt_id"]
    assert len(s2["log"]) == 2


def test_stage3_grind_bandit_9_linkage_metadata_never_changes_the_model(
        iso_state):
    # Run A: an attempt logged against an explicit recommendation id.
    cli.main(["next", "--seed", "7"])
    rec = st.load()["recommendations"][0]
    pid = rec["selected_problem_id"]
    cli.main(["log", str(pid), "solved", "--recommendation-id", rec["id"]])
    a = st.load()
    # Run B: the same pick and the same attempt with no linkage at all.
    iso_state.unlink()
    cli.main(["next", "--seed", "7"])
    cli.main(["log", str(pid), "solved"])
    b = st.load()
    # State-level record_attempt without any linkage metadata.
    s = _blank_state()
    bandit.observe(s, np.zeros(bandit.D), 0.25)
    plain = copy.deepcopy(s)
    st.record_attempt(plain, _problem(1), True, 20, 0.25, T0)
    linked = copy.deepcopy(s)
    st.record_attempt(linked, _problem(1), True, 20, 0.25, T0,
                      recommendation_id="rec-x", linkage_method="explicit",
                      recommendation_status={"exposed": True,
                                             "accepted": True})
    assert plain["bandit"] == linked["bandit"]
    assert plain["topics"] == linked["topics"]
    assert plain["problems"] == linked["problems"]
    ea, eb = plain["log"][-1], linked["log"][-1]
    assert ea["reward"] == eb["reward"] == 0.25
    assert "recommendation_id" not in ea
    assert eb["recommendation_id"] == "rec-x"
    # The CLI runs agree on the model state too. Wall-clock fields
    # (topic timestamps, problem last_ts) legitimately differ between
    # the two runs, so compare the learned quantities themselves.
    assert np.allclose(a["bandit"]["mean"], b["bandit"]["mean"])
    assert np.allclose(a["bandit"]["prec"], b["bandit"]["prec"])
    for t in a["topics"]:
        ka = {k: a["topics"][t][k] for k in ("alpha", "beta", "attempts", "solved")}
        kb = {k: b["topics"][t][k] for k in ("alpha", "beta", "attempts", "solved")}
        assert ka == kb
    assert a["log"][-1]["reward"] == b["log"][-1]["reward"]
