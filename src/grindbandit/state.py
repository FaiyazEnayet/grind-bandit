"""Local persistence. Everything lives in one JSON file.

Besides topics, problems, the attempt log, and the bandit posterior, the
state holds two append-only records: `recommendations` (one full record
per pick, so any recommendation can be reconstructed and audited) and
`delayed_quizzes` (delayed transfer quiz schedules and their outcomes).
Delayed quiz outcomes are observational records: they live only in the
quiz store and are never fed into mastery, the bandit posterior, or the
model-derived reward on attempt entries.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

# Days between logging an attempt and its delayed transfer quiz coming due.
DELAYED_QUIZ_DELAY_DAYS = 3


def default_path() -> str:
    return os.environ.get(
        "GRIND_BANDIT_STATE", str(Path.home() / ".grind-bandit" / "state.json")
    )


def load(path: str | None = None) -> dict:
    path = path or default_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "topics": {},
        "problems": {},
        "log": [],
        "bandit": {},
        "recommendations": [],
        "delayed_quizzes": [],
    }


def save(state: dict, path: str | None = None) -> str:
    path = path or default_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=1)
    return path


def record_attempt(
    state: dict,
    problem: dict,
    solved: bool,
    minutes: int,
    reward: float,
    now: float,
    recommendation_id: str | None = None,
) -> None:
    entry = {
        "ts": now,
        "problem_id": problem["id"],
        "title": problem["title"],
        "difficulty": problem["difficulty"],
        "topics": problem["topics"],
        "solved": solved,
        "minutes": minutes,
        "reward": reward,
        # The reward is computed by the mastery model itself (change in
        # modeled mastery of the weakest topic). It is labeled as such so
        # it is never mistaken for a measured learning outcome.
        "reward_kind": "model_derived_mastery_change",
    }
    if recommendation_id is not None:
        entry["recommendation_id"] = recommendation_id
    state.setdefault("log", []).append(entry)
    p = state.setdefault("problems", {}).setdefault(
        str(problem["id"]),
        {"attempts": 0, "solved": 0, "last_ts": None, "ever_solved": False},
    )
    p["attempts"] += 1
    p["last_ts"] = now
    if solved:
        p["solved"] += 1
        p["ever_solved"] = True


def append_recommendation(state: dict, record: dict) -> None:
    state.setdefault("recommendations", []).append(record)


def latest_recommendation_id(state: dict, problem_id: int) -> str | None:
    """Id of the most recent recommendation that selected this problem."""
    for rec in reversed(state.get("recommendations", [])):
        if rec.get("selected_problem_id") == problem_id:
            return rec.get("id")
    return None


def _assigned_quiz_problem_ids(state: dict) -> list:
    """Problem ids assigned to earlier quizzes, in assignment order.

    Together with the quiz records themselves this is the exposure
    history: how many times a problem has been handed out as a quiz is
    recoverable from the store, and each new record also carries its
    own count (see schedule_delayed_quiz).
    """
    return [q.get("quiz_problem_id") for q in state.get("delayed_quizzes", [])]


def _quiz_pool(bank: list[dict], attempted: dict, topic: str) -> list[dict]:
    return [p for p in bank if topic in p["topics"] and p["id"] != attempted["id"]]


def _is_novel(problem: dict, attempted_ids: set, assigned_ids: set) -> bool:
    """A problem is novel as a quiz item if it was never attempted and
    never assigned to an earlier quiz."""
    return str(problem["id"]) not in attempted_ids and problem["id"] not in assigned_ids


def choose_quiz_problem(
    bank: list[dict], attempted: dict, topic: str, state: dict
) -> dict | None:
    """Pick a different problem on the same topic for a delayed quiz.

    Deterministic priority, lowest id inside each tier:

    1. novel problems: never attempted and never assigned as a quiz;
    2. assigned before but never attempted;
    3. attempted before but never assigned;
    4. attempted and assigned before.

    A problem is therefore never assigned twice while a novel
    alternative exists, and previously attempted problems are the last
    resort. When only reused problems remain the pick is still made;
    schedule_delayed_quiz states that exhaustion on the record.
    """
    pool = _quiz_pool(bank, attempted, topic)
    if not pool:
        return None
    attempted_ids = set(state.get("problems", {}).keys())
    assigned_ids = {i for i in _assigned_quiz_problem_ids(state) if i is not None}

    def tier(p: dict) -> tuple:
        return (str(p["id"]) in attempted_ids, p["id"] in assigned_ids)

    return min(pool, key=lambda p: (tier(p), p["id"]))


def schedule_delayed_quiz(
    state: dict,
    bank: list[dict],
    attempted: dict,
    topic: str,
    now: float,
    recommendation_id: str | None = None,
) -> dict | None:
    """Schedule a delayed transfer quiz for an attempt. Returns it or None.

    The quiz targets the attempt's weakest topic with a different
    problem and comes due DELAYED_QUIZ_DELAY_DAYS later. It is only a
    schedule: the outcome, when recorded, lands in this same store and
    nowhere else.
    """
    quiz_problem = choose_quiz_problem(bank, attempted, topic, state)
    if quiz_problem is None:
        return None
    assigned_ids = _assigned_quiz_problem_ids(state)
    attempted_ids = set(state.get("problems", {}).keys())
    novel_pool = [
        p for p in _quiz_pool(bank, attempted, topic)
        if _is_novel(p, attempted_ids, set(assigned_ids))
    ]
    quiz = {
        "id": f"quiz-{uuid.uuid4().hex[:12]}",
        "kind": "delayed_transfer_quiz",
        "scheduled_ts": now,
        "due_ts": now + DELAYED_QUIZ_DELAY_DAYS * 86400.0,
        "delay_days": DELAYED_QUIZ_DELAY_DAYS,
        "topic": topic,
        "source_problem_id": attempted["id"],
        "source_recommendation_id": recommendation_id,
        "quiz_problem_id": quiz_problem["id"],
        "quiz_title": quiz_problem["title"],
        # How this problem was chosen: whether it is novel (never
        # attempted, never assigned), how many times it was assigned
        # to earlier quizzes, and whether the novel pool was already
        # exhausted, in which case this assignment is an explicit
        # reuse rather than a silent repeat.
        "assignment": {
            "novel": _is_novel(quiz_problem, attempted_ids, set(assigned_ids)),
            "prior_assignments": assigned_ids.count(quiz_problem["id"]),
            "pool_exhausted": not novel_pool,
        },
        "status": "pending",
        "outcome": None,
    }
    state.setdefault("delayed_quizzes", []).append(quiz)
    return quiz


def find_delayed_quiz(state: dict, quiz_id: str) -> dict | None:
    for q in state.get("delayed_quizzes", []):
        if q.get("id") == quiz_id:
            return q
    return None


def record_delayed_outcome(
    state: dict, quiz_id: str, solved: bool, now: float, allow_early: bool = False
) -> dict | None:
    """Store a delayed quiz outcome in the quiz store only.

    Deliberately touches nothing else: the outcome is an observational
    record, kept separate from attempts, mastery, and the bandit
    posterior, and it never feeds the model-derived reward.

    The quiz measures retention after its delay, so an outcome dated
    before the due date is rejected with a ValueError unless
    allow_early is passed explicitly. A record with no due date (none
    are written that way anymore) is treated as due. The stored
    outcome carries the realized delay in seconds (scheduled to
    recorded) and an early / on-time / late status: early is before
    the due date, on-time is the due day itself, late is after it.
    """
    quiz = find_delayed_quiz(state, quiz_id)
    if quiz is None or quiz.get("status") != "pending":
        return None
    due_ts = quiz.get("due_ts")
    if due_ts is not None and now < due_ts and not allow_early:
        raise ValueError(
            f"delayed quiz {quiz_id} has not come due yet; an outcome "
            "recorded before the due date does not measure retention "
            "after the delay (pass allow_early=True to record it anyway)"
        )
    scheduled_ts = quiz.get("scheduled_ts")
    if due_ts is not None and now < due_ts:
        timing = "early"
    elif due_ts is not None and now > due_ts + 86400.0:
        timing = "late"
    else:
        timing = "on_time"
    quiz["status"] = "completed"
    quiz["outcome"] = {
        "solved": bool(solved),
        "recorded_ts": now,
        "delay_seconds": None if scheduled_ts is None else now - scheduled_ts,
        "timing": timing,
    }
    return quiz


def pending_delayed_quizzes(state: dict) -> list[dict]:
    return [q for q in state.get("delayed_quizzes", []) if q.get("status") == "pending"]
