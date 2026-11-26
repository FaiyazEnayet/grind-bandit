"""Offline-only credit-assignment audit for per-topic mastery.

``mastery.update`` credits every tagged topic equally: a solved attempt on
a problem tagged [array, hash-table] adds one win to both topics, even
when the solution may have relied mostly on one of them. 190 of the 218
curated problems carry multiple tags, so the choice of credit rule moves
the mastery picture.

This module replays the same logged attempts under three rules and
reports the per-topic differences:

- ``full``: every tagged topic gets full credit (the live rule).
- ``fractional``: credit is split evenly, 1/n per tagged topic.
- ``blocked``: only the user-reported blocked topic gets credit. An
  attempt that names no blocked topic falls back to fractional credit
  for that attempt, and the fallback is counted in the report.

Nothing here touches live state. The audit builds its own fresh state
from the entries it is given, never mutates those entries, and never
feeds the bandit or the reward. Results are counterfactual modeling of
one log under different assumptions, not evidence that any rule learns
better. The blocked rule treats a subjective user annotation as a
modeling input, never as ground truth.
"""

from __future__ import annotations

import math

from grindbandit import mastery

RULES = ("full", "fractional", "blocked")

LABEL = (
    "counterfactual modeling: the same log replayed under different "
    "credit-assignment assumptions; not a learning improvement"
)

ASSUMPTIONS = {
    "full": (
        "every tagged topic is treated as equally responsible for the "
        "outcome; this is the live rule"
    ),
    "fractional": (
        "responsibility is split evenly across the tagged topics, 1/n each"
    ),
    "blocked": (
        "only the user-reported blocked topic is treated as responsible; "
        "a user report is a subjective modeling input, never ground truth"
    ),
}


def _topic(state: dict, name: str) -> dict:
    topics = state.setdefault("topics", {})
    return topics.setdefault(
        name, {"alpha": 1.0, "beta": 1.0, "attempts": 0, "solved": 0, "last_ts": None}
    )


def _validate_entry(entry: dict) -> dict:
    """Normalize one log entry, or raise ValueError on invalid input.

    Returns a fresh dict, so the caller's entry is never modified.
    Duplicate tags are de-duplicated in order; anything else malformed
    is rejected rather than guessed at.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"attempt entry must be a mapping, got {type(entry).__name__}")
    topics = entry.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("attempt entry needs a non-empty list of topics")
    clean: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        if not isinstance(topic, str) or not topic:
            raise ValueError(f"topic names must be non-empty strings, got {topic!r}")
        if topic not in seen:
            seen.add(topic)
            clean.append(topic)
    solved = entry.get("solved")
    if not isinstance(solved, bool):
        raise ValueError(f"solved must be a bool, got {solved!r}")
    ts = entry.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
        raise ValueError(f"ts must be a finite number, got {ts!r}")
    blocked = entry.get("blocked_topic")
    if blocked is not None:
        if not isinstance(blocked, str) or not blocked:
            raise ValueError(f"blocked_topic must be a non-empty string, got {blocked!r}")
        if blocked not in clean:
            raise ValueError(
                f"blocked topic {blocked!r} is not one of the attempt tags {clean}"
            )
    return {"ts": ts, "topics": clean, "solved": solved, "blocked_topic": blocked}


def _weights(topics: list[str], blocked_topic: str | None, rule: str) -> tuple[list[float], bool]:
    """Per-topic credit weights for one attempt under a rule.

    Returns the weights and whether the blocked rule fell back to
    fractional credit because the attempt names no blocked topic.
    """
    n = len(topics)
    if rule == "full":
        return [1.0] * n, False
    if rule == "fractional":
        return [1.0 / n] * n, False
    if blocked_topic is not None:
        return [1.0 if topic == blocked_topic else 0.0 for topic in topics], False
    return [1.0 / n] * n, True


def _apply(state: dict, topics: list[str], solved: bool, now: float,
           weights: list[float]) -> None:
    """Apply one attempt with per-topic weights.

    Mirrors ``mastery.update`` exactly when every weight is 1.0: decay
    is persisted up to ``now`` before the new observation lands, so a
    new attempt can never revive long-forgotten evidence. Topics with a
    zero weight are left untouched: no credit, no decay-clock reset.
    """
    for name, weight in zip(topics, weights):
        if weight == 0.0:
            continue
        topic = _topic(state, name)
        alpha, beta = mastery.decayed_counts(state, name, now)
        topic["alpha"], topic["beta"] = alpha, beta
        if solved:
            topic["alpha"] += weight
            topic["solved"] += 1
        else:
            topic["beta"] += weight
        topic["attempts"] += 1
        topic["last_ts"] = now


def replay_under_rule(entries: list[dict], upto_ts: float, rule: str,
                      half_life_days: float | None = None) -> dict:
    """Replay a log under one credit rule.

    Returns the mastery map, the effective per-topic counts as of
    ``upto_ts`` (alpha, beta, attempts, solved; untouched topics show
    the 1.0/1.0 prior with zero attempts), the rule's assumption, and
    replay metadata. The entries are validated and copied, never
    mutated, and attempts after ``upto_ts`` are ignored.
    """
    if rule not in RULES:
        raise ValueError(f"unknown credit rule {rule!r}; expected one of {RULES}")
    if (isinstance(upto_ts, bool) or not isinstance(upto_ts, (int, float))
            or not math.isfinite(upto_ts)):
        raise ValueError(f"upto_ts must be a finite number, got {upto_ts!r}")
    clean = [_validate_entry(entry) for entry in entries]
    fake: dict = {}
    if half_life_days is not None:
        fake["config"] = {"half_life_days": half_life_days}
    seen: set[str] = set()
    n_attempts = 0
    n_multi_tag = 0
    fallbacks = 0
    for entry in clean:
        if entry["ts"] > upto_ts:
            continue
        seen.update(entry["topics"])
        weights, fell_back = _weights(entry["topics"], entry["blocked_topic"], rule)
        if fell_back:
            fallbacks += 1
        _apply(fake, entry["topics"], entry["solved"], entry["ts"], weights)
        n_attempts += 1
        if len(entry["topics"]) > 1:
            n_multi_tag += 1
    mastery_map = {
        name: mastery.mean(fake, name, upto_ts) for name in sorted(seen)
    }
    counts = {}
    for name in sorted(seen):
        topic = fake.get("topics", {}).get(name)
        if topic is None:
            counts[name] = {"alpha": 1.0, "beta": 1.0, "attempts": 0, "solved": 0}
        else:
            alpha, beta = mastery.decayed_counts(fake, name, upto_ts)
            counts[name] = {
                "alpha": alpha,
                "beta": beta,
                "attempts": topic["attempts"],
                "solved": topic["solved"],
            }
    return {
        "rule": rule,
        "assumption": ASSUMPTIONS[rule],
        "mastery": mastery_map,
        "counts": counts,
        "meta": {
            "n_attempts": n_attempts,
            "n_multi_tag_attempts": n_multi_tag,
            "blocked_fallbacks": fallbacks,
        },
    }


def compare_rules(entries: list[dict], upto_ts: float,
                  half_life_days: float | None = None,
                  rules: tuple[str, ...] = RULES) -> dict:
    """Replay the same log under each credit rule and compare by topic.

    The top-level label states plainly that this is counterfactual
    modeling, not a learning improvement. Per-topic differences carry
    the mastery under each rule plus the max spread, and every rule
    carries its assumption and its effective counts, so a large spread
    on a barely practiced topic is visible as such.
    """
    unknown = [rule for rule in rules if rule not in RULES]
    if unknown:
        raise ValueError(f"unknown credit rule(s) {unknown}; expected one of {RULES}")
    replays = {
        rule: replay_under_rule(entries, upto_ts, rule, half_life_days)
        for rule in rules
    }
    topics: set[str] = set()
    for replay in replays.values():
        topics.update(replay["mastery"])
    differences: dict[str, dict] = {}
    for topic in sorted(topics):
        values = {
            rule: replays[rule]["mastery"].get(topic, 0.5) for rule in rules
        }
        values["max_spread"] = max(values.values()) - min(values.values())
        differences[topic] = values
    max_topic = None
    max_spread = 0.0
    if differences:
        max_topic = max(differences, key=lambda t: differences[t]["max_spread"])
        max_spread = differences[max_topic]["max_spread"]
    first = next(iter(replays.values())) if replays else None
    return {
        "label": LABEL,
        "upto_ts": upto_ts,
        "n_attempts": first["meta"]["n_attempts"] if first else 0,
        "n_multi_tag_attempts": first["meta"]["n_multi_tag_attempts"] if first else 0,
        "rules": replays,
        "differences": differences,
        "max_spread_topic": max_topic,
        "max_spread": max_spread,
    }
