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


def choose_quiz_problem(
    bank: list[dict], attempted: dict, topic: str, state: dict
) -> dict | None:
    """Pick a different problem on the same topic for a delayed quiz.

    Prefers problems with no recorded attempt; ties break to the lowest
    id, so the choice is deterministic for a given bank and state.
    """
    pool = [p for p in bank if topic in p["topics"] and p["id"] != attempted["id"]]
    if not pool:
        return None
    attempted_ids = set(state.get("problems", {}).keys())
    fresh = [p for p in pool if str(p["id"]) not in attempted_ids]
    return min(fresh or pool, key=lambda p: p["id"])


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
    state: dict, quiz_id: str, solved: bool, now: float
) -> dict | None:
    """Store a delayed quiz outcome in the quiz store only.

    Deliberately touches nothing else: the outcome is an observational
    record, kept separate from attempts, mastery, and the bandit
    posterior, and it never feeds the model-derived reward.
    """
    quiz = find_delayed_quiz(state, quiz_id)
    if quiz is None or quiz.get("status") != "pending":
        return None
    quiz["status"] = "completed"
    quiz["outcome"] = {"solved": bool(solved), "recorded_ts": now}
    return quiz


def pending_delayed_quizzes(state: dict) -> list[dict]:
    return [q for q in state.get("delayed_quizzes", []) if q.get("status") == "pending"]
