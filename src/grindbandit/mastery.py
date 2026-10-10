"""Per-topic skill as a Beta Bernoulli model.

Each topic holds alpha (wins) and beta (losses). Mastery is the posterior
mean alpha / (alpha + beta). A solved attempt adds one win to every topic
on the problem, a failed attempt adds one loss. New topics start at
alpha = beta = 1, which is a mastery of 50 percent: maximum uncertainty.
"""

from __future__ import annotations


def _topic(state: dict, name: str) -> dict:
    topics = state.setdefault("topics", {})
    return topics.setdefault(
        name, {"alpha": 1.0, "beta": 1.0, "attempts": 0, "solved": 0, "last_ts": None}
    )


def mean(state: dict, name: str) -> float:
    t = _topic(state, name)
    return t["alpha"] / (t["alpha"] + t["beta"])


def mastery_map(state: dict) -> dict[str, float]:
    return {name: mean(state, name) for name in state.get("topics", {})}


def update(state: dict, topics: list[str], solved: bool, now: float) -> None:
    for name in topics:
        t = _topic(state, name)
        if solved:
            t["alpha"] += 1.0
            t["solved"] += 1
        else:
            t["beta"] += 1.0
        t["attempts"] += 1
        t["last_ts"] = now


def replay(entries: list[dict], upto_ts: float) -> dict[str, float]:
    """Rebuild per-topic mastery from the attempt log up to a timestamp.

    This powers the weekly review: replaying the log gives an honest
    picture of what mastery looked like at any point in the past.
    """
    fake: dict = {}
    for e in entries:
        if e["ts"] <= upto_ts:
            update(fake, e["topics"], e["solved"], e["ts"])
    return mastery_map(fake)
