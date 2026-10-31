"""Shared demo-data builder for the dashboard and GIF generators.

Builds the scripted 30-attempt history with the real grindbandit engine:
a seeded user who solves 90% of array/string problems, 30% of
dp/graph/backtracking, and 70% of everything else, over 13 days.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grindbandit import bandit, mastery, problems  # noqa: E402
from grindbandit import state as st  # noqa: E402

BANK = problems.load()
TOPICS = problems.all_topics(BANK)
DAY = 86400.0


def log_attempt(state: dict, problem: dict, solved: bool, minutes: int, now: float):
    """One attempt through the real engine. Same steps as cli.cmd_log."""
    x = bandit.context(problem, state, now)
    tm = {t: mastery.mean(state, t, now) for t in problem["topics"]}
    weak = bandit.weakest_topic(problem, tm)
    before = tm[weak]
    mastery.update(state, problem["topics"], solved, now)
    after = mastery.mean(state, weak, now)
    reward = max(-1.0, min(1.0, after - before))
    bandit.observe(state, x, reward)
    st.record_attempt(state, problem, solved, minutes, reward, now)
    return reward, weak, after


def build_demo_state(seed: int = 42) -> dict:
    state = {"topics": {}, "problems": {}, "log": [], "bandit": {}}
    rng = random.Random(seed)
    start = time.time() - 13 * DAY
    strong = {"array", "string"}
    weak = {"dp", "graph", "backtracking"}
    order = BANK[:]
    rng.shuffle(order)
    for n, p in enumerate(order[:30]):
        topics = set(p["topics"])
        rate = 0.9 if topics & strong else (0.3 if topics & weak else 0.7)
        solved = rng.random() < rate
        ts = start + n * (13 * DAY / 30)
        log_attempt(state, p, solved, rng.randint(15, 60), ts)
    return state


def recommend(state: dict, seed: int | None = None, now: float | None = None):
    """Next-problem pick with real Thompson sampling. Mirrors cli.cmd_next."""
    from grindbandit import cli

    now = time.time() if now is None else now
    cands, review_mode = cli.select_candidates(BANK, state, now)
    theta = bandit.sample_theta(state, seed=seed)
    scored = sorted(
        ((bandit.score(theta, bandit.context(p, state, now)), p["id"], p) for p in cands),
        key=lambda t: (-t[0], t[1]),
    )
    _, _, p = scored[0]
    return p, bandit.explain(p, state, now, theta), review_mode
