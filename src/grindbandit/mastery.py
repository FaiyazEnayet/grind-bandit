"""Per-topic skill as a Beta Bernoulli model with forgetting.

Each topic holds alpha (wins) and beta (losses). Mastery is the posterior
mean alpha / (alpha + beta). A solved attempt adds one win to every topic
on the problem, a failed attempt adds one loss. New topics start at
alpha = beta = 1, which is a mastery of 50 percent: maximum uncertainty.

Forgetting: on read, the stored counts decay exponentially toward the
(1, 1) prior based on time since the topic was last attempted. The
half-life defaults to 30 days and is configurable via
state["config"]["half_life_days"]. Decay is applied on read only; the
stored counts are never mutated, so the raw evidence stays intact.
"""

from __future__ import annotations

HALF_LIFE_DAYS = 30.0
_SECONDS_PER_DAY = 86400.0


def _topic(state: dict, name: str) -> dict:
    topics = state.setdefault("topics", {})
    return topics.setdefault(
        name, {"alpha": 1.0, "beta": 1.0, "attempts": 0, "solved": 0, "last_ts": None}
    )


def half_life_days(state: dict) -> float:
    return float(state.get("config", {}).get("half_life_days", HALF_LIFE_DAYS))


def decayed_counts(state: dict, name: str, now: float | None) -> tuple[float, float]:
    """Effective (alpha, beta) after decaying toward the prior.

    With w = 0.5 ** (days_elapsed / half_life), the effective counts are
    1 + (alpha - 1) * w and 1 + (beta - 1) * w. No time passed (or no
    timestamp yet) means w = 1: no decay. Far future means w -> 0:
    full return to the 50 percent prior.
    """
    t = _topic(state, name)
    last = t.get("last_ts")
    if now is None or last is None or now <= last:
        return t["alpha"], t["beta"]
    w = 0.5 ** ((now - last) / _SECONDS_PER_DAY / half_life_days(state))
    return 1.0 + (t["alpha"] - 1.0) * w, 1.0 + (t["beta"] - 1.0) * w


def mean(state: dict, name: str, now: float | None = None) -> float:
    a, b = decayed_counts(state, name, now)
    return a / (a + b)


def mastery_map(state: dict, now: float | None = None) -> dict[str, float]:
    return {name: mean(state, name, now) for name in state.get("topics", {})}


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
    Forgetting is applied as of upto_ts, so a topic untouched for weeks
    before that point shows decayed, not frozen, mastery.
    """
    fake: dict = {}
    for e in entries:
        if e["ts"] <= upto_ts:
            update(fake, e["topics"], e["solved"], e["ts"])
    return mastery_map(fake, upto_ts)
