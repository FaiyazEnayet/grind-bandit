"""Local persistence. Everything lives in one JSON file."""

from __future__ import annotations

import json
import os
from pathlib import Path


def default_path() -> str:
    return os.environ.get(
        "GRIND_BANDIT_STATE", str(Path.home() / ".grind-bandit" / "state.json")
    )


def load(path: str | None = None) -> dict:
    path = path or default_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"topics": {}, "problems": {}, "log": [], "bandit": {}}


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
    }
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
