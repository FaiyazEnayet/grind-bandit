"""GB8: eligibility classes and spaced-review resurfacing.

Eligibility is an explicit policy with three classes: new practice,
due review (a solved problem whose review interval has elapsed), and
cooldown (attempted too recently in either class). Every problem in
the bank gets a deterministic eligibility reason, the recommendation
record carries the policy and the per-problem eligibility it used,
and an empty eligible set produces an explicit message with no record
appended. Recency is the days since the least recently practiced
topic on the problem, with never-practiced topics counting as the
30-day cap; context-v1 records keep replaying under the preserved v1
semantics.
"""

import sys
from pathlib import Path

import pytest

from grindbandit import bandit, cli, problems, state as st

DAY = 86400.0
BASE = 1_700_000_000.0


@pytest.fixture
def iso_state(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    monkeypatch.setenv("GRIND_BANDIT_STATE", str(p))
    return p


def _solved_record(ts):
    return {"attempts": 1, "solved": 1, "last_ts": ts, "ever_solved": True}


def _attempted_record(ts):
    return {"attempts": 1, "solved": 0, "last_ts": ts, "ever_solved": False}


def _entry(entries, pid):
    return next(e for e in entries if e["problem_id"] == pid)


def _bank():
    return problems.load()


# --- due review resurfacing -------------------------------------------------

def test_stage3_grind_bandit_8_solved_problem_reappears_when_review_due():
    bank = _bank()
    state = {"problems": {"1": _solved_record(BASE)}}
    entries = cli.assess_eligibility(bank, state, BASE + 8 * DAY)
    e = _entry(entries, 1)
    assert e["eligible"] is True
    assert e["class"] == "due_review"
    assert "due" in e["reason"]
    cands, review_mode = cli.select_candidates(bank, state, BASE + 8 * DAY)
    assert 1 in {p["id"] for p in cands}
    assert review_mode is True


def test_stage3_grind_bandit_8_solved_problem_stays_out_before_due():
    bank = _bank()
    state = {"problems": {"1": _solved_record(BASE)}}
    entries = cli.assess_eligibility(bank, state, BASE + 3 * DAY)
    e = _entry(entries, 1)
    assert e["eligible"] is False
    assert e["class"] == "review_cooldown"
    cands, review_mode = cli.select_candidates(bank, state, BASE + 3 * DAY)
    assert 1 not in {p["id"] for p in cands}
    assert review_mode is False


def test_stage3_grind_bandit_8_review_interval_comes_from_state_config():
    bank = _bank()
    state = {
        "problems": {"1": _solved_record(BASE)},
        "config": {"review_interval_days": 2.0},
    }
    assert bandit.review_interval_days(state) == 2.0
    assert bandit.review_interval_days({}) == bandit.REVIEW_INTERVAL_DAYS
    e = _entry(cli.assess_eligibility(bank, state, BASE + 3 * DAY), 1)
    assert e["class"] == "due_review" and e["eligible"] is True


# --- cooldown ----------------------------------------------------------------

def test_stage3_grind_bandit_8_cooldown_excludes_recent_attempts():
    bank = _bank()
    now = BASE + 10 * DAY
    state = {"problems": {
        "1": _attempted_record(now - 12 * 3600.0),
        "2": _attempted_record(now - 2 * DAY),
        "3": _solved_record(now - 3600.0),
    }}
    entries = cli.assess_eligibility(bank, state, now)
    e1, e2, e3 = _entry(entries, 1), _entry(entries, 2), _entry(entries, 3)
    assert e1["eligible"] is False and e1["class"] == "cooldown_new_practice"
    assert e2["eligible"] is True and e2["class"] == "new_practice"
    # Solved an hour ago: inside both the attempt cooldown and the
    # review interval; the sooner gate is reported.
    assert e3["eligible"] is False and e3["class"] == "review_cooldown"
    cands, _ = cli.select_candidates(bank, state, now)
    ids = {p["id"] for p in cands}
    assert 1 not in ids and 3 not in ids and 2 in ids


def test_stage3_grind_bandit_8_every_problem_gets_a_deterministic_reason():
    bank = _bank()
    now = BASE + 10 * DAY
    state = {"problems": {
        "1": _solved_record(BASE),
        "2": _attempted_record(now - 3600.0),
        "3": _solved_record(now - 3600.0),
    }}
    first = cli.assess_eligibility(bank, state, now)
    second = cli.assess_eligibility(bank, state, now)
    assert first == second
    assert len(first) == len(bank)
    for e in first:
        assert e["class"] in {
            "new_practice", "due_review",
            "cooldown_new_practice", "review_cooldown",
        }
        assert e["reason"]


# --- empty eligible set --------------------------------------------------------

def test_stage3_grind_bandit_8_empty_set_appends_no_record_and_says_so(
    iso_state, capsys, monkeypatch
):
    now = BASE + 10 * DAY
    bank = _bank()
    s = st.load()
    for p in bank:
        s["problems"][str(p["id"])] = _attempted_record(now - 3600.0)
    st.save(s)
    monkeypatch.setattr(cli, "_now", lambda: now)
    cli.main(["next", "--seed", "1"])
    out = capsys.readouterr().out
    assert "No problem is eligible right now" in out
    # The old code claimed this situation meant everything was solved.
    assert "solved every problem" not in out
    assert "Review mode" not in out
    assert st.load().get("recommendations", []) == []


# --- logged policy metadata -----------------------------------------------------

def test_stage3_grind_bandit_8_logged_eligible_set_matches_policy(
    iso_state, capsys, monkeypatch
):
    now = BASE + 8 * DAY
    bank = _bank()
    s = st.load()
    s["problems"]["1"] = _solved_record(BASE)
    for p in bank:
        if p["id"] != 1:
            s["problems"][str(p["id"])] = _attempted_record(now - 3600.0)
    st.save(s)
    monkeypatch.setattr(cli, "_now", lambda: now)
    cli.main(["next", "--seed", "4"])
    out = capsys.readouterr().out
    assert "review interval has elapsed" in out
    rec = st.load()["recommendations"][-1]
    assert rec["feature_version"] == "context-v2"
    assert rec["review_mode"] is True
    assert rec["selected_problem_id"] == 1
    policy = rec["policy"]
    assert policy["review_interval_days"] == bandit.REVIEW_INTERVAL_DAYS
    assert policy["feature_version"] == "context-v2"
    logged_eligible = {
        e["problem_id"] for e in policy["eligibility"] if e["eligible"]
    }
    assert logged_eligible == {1}
    assert {c["problem_id"] for c in rec["candidates"]} == logged_eligible
    by_id = {e["problem_id"]: e for e in policy["eligibility"]}
    assert by_id[1]["class"] == "due_review"
    cand = rec["candidates"][0]
    assert cand["eligibility_class"] == "due_review"
    assert cand["eligibility_reason"] == by_id[1]["reason"]
    assert bandit.replay_selection(rec) == 1


# --- recency: documented stale-tag interpretation ------------------------------

def _topic(last_ts, alpha=3.0, beta=1.0):
    return {"alpha": alpha, "beta": beta, "attempts": 1,
            "solved": 1, "last_ts": last_ts}


def test_stage3_grind_bandit_8_recency_uses_least_recently_practiced_topic():
    now = BASE + 20 * DAY
    state = {"topics": {
        "array": _topic(BASE),               # 20 days ago
        "dp": _topic(BASE + 18 * DAY),       # 2 days ago
    }}
    p = {"id": 5, "title": "Mixed", "difficulty": "Medium",
         "topics": ["array", "dp"]}
    x = bandit.context(p, state, now)
    assert x[3] == pytest.approx(20.0 / 30.0)
    # The v1 semantics took the most recent practice instead.
    x1 = bandit.context_v1(p, state, now)
    assert x1[3] == pytest.approx(2.0 / 30.0)
    assert x[3] != x1[3]


def test_stage3_grind_bandit_8_recency_counts_unseen_topic_as_cap():
    now = BASE + 20 * DAY
    state = {"topics": {"array": _topic(BASE + 18 * DAY)}}
    mixed = {"id": 6, "title": "HalfNew", "difficulty": "Easy",
             "topics": ["array", "dp"]}
    assert bandit.context(mixed, state, now)[3] == 1.0
    assert bandit.context_v1(mixed, state, now)[3] == pytest.approx(2.0 / 30.0)
    fresh = {"id": 7, "title": "Fresh", "difficulty": "Easy", "topics": ["dp"]}
    assert bandit.context(fresh, state, now)[3] == 1.0
    assert bandit.context_v1(fresh, state, now)[3] == 1.0


# --- replay across feature versions ---------------------------------------------

def _record_with_contexts(state, problems_with_x, seed):
    import numpy as np

    mean, prec = bandit.get_posterior(state)
    theta = bandit.sample_theta(state, seed=seed)
    scored = [(bandit.score(theta, x), p) for p, x in problems_with_x]
    winner = min(scored, key=lambda t: (-t[0], t[1]["id"]))[1]
    return theta, winner


def test_stage3_grind_bandit_8_v1_record_still_replays():
    now = BASE + 20 * DAY
    state = {"topics": {
        "array": _topic(BASE),
        "dp": _topic(BASE + 18 * DAY),
        "graph": _topic(BASE + 10 * DAY),
    }, "problems": {}, "log": [], "bandit": {}}
    pa = {"id": 5, "title": "Mixed", "difficulty": "Medium",
          "topics": ["array", "dp"]}
    pb = {"id": 9, "title": "Graphy", "difficulty": "Easy", "topics": ["graph"]}
    xa = bandit.context_v1(pa, state, now)
    xb = bandit.context_v1(pb, state, now)
    _, winner = _record_with_contexts(state, [(pa, xa), (pb, xb)], seed=17)
    mean, prec = bandit.get_posterior(state)
    rec = {
        "id": "rec-v1fixture",
        "ts": now,
        "feature_version": "context-v1",
        "seed": 17,
        "posterior": {"mean": mean.tolist(), "prec": prec.tolist()},
        "candidates": [
            {"problem_id": 5, "context": [float(v) for v in xa]},
            {"problem_id": 9, "context": [float(v) for v in xb]},
        ],
        "selected_problem_id": winner["id"],
    }
    assert bandit.replay_selection(rec) == winner["id"]
    assert bandit.replay_selection_v1(rec) == winner["id"]


def test_stage3_grind_bandit_8_v2_record_replays_and_version_gate_holds():
    state = {"topics": {}, "problems": {}, "log": [], "bandit": {}}
    pa = {"id": 5, "title": "A", "difficulty": "Easy", "topics": ["array"]}
    pb = {"id": 9, "title": "B", "difficulty": "Hard", "topics": ["dp"]}
    rec = bandit.build_recommendation(state, [pa, pb], BASE, seed=3)
    assert rec["feature_version"] == "context-v2"
    assert bandit.replay_selection(rec) == rec["selected_problem_id"]
    bad = dict(rec, feature_version="context-v999")
    with pytest.raises(ValueError):
        bandit.replay_selection(bad)


# --- old state loads ---------------------------------------------------------

def test_stage3_grind_bandit_8_legacy_state_without_policy_fields_loads(
    iso_state, capsys
):
    import json

    legacy = {"topics": {}, "problems": {}, "log": [], "bandit": {}}
    iso_state.write_text(json.dumps(legacy))
    cli.main(["next", "--seed", "2"])
    out = capsys.readouterr().out
    assert "Why this one:" in out
    rec = st.load()["recommendations"][-1]
    assert rec["policy"]["review_interval_days"] == bandit.REVIEW_INTERVAL_DAYS


# --- scripts/common follows the same policy -------------------------------------

def test_stage3_grind_bandit_8_scripts_common_recommend_uses_same_policy():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import common

    now = BASE + 8 * DAY
    bank = common.BANK
    empty = {"topics": {}, "log": [], "bandit": {}, "problems": {
        str(p["id"]): _attempted_record(now - 3600.0) for p in bank
    }}
    p, lines, review_mode = common.recommend(empty, seed=1, now=now)
    assert p is None and lines == [] and review_mode is False

    due = {"topics": {}, "log": [], "bandit": {}, "problems": {
        str(p["id"]): _attempted_record(now - 3600.0) for p in bank
    }}
    due["problems"]["1"] = _solved_record(BASE)
    p, lines, review_mode = common.recommend(due, seed=1, now=now)
    assert p["id"] == 1 and review_mode is True and lines
