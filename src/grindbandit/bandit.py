"""Thompson sampling for a linear bandit over practice problems.

The model: expected learning gain from a problem is linear in a small
context vector, reward = theta . x. Theta has a Gaussian posterior that
starts at zero (no assumptions) and updates with every logged attempt.
To pick, we sample one theta from the posterior and take the argmax.
Early on the posterior is wide, so sampling explores. As evidence
accumulates it concentrates, so sampling exploits. That is the whole
explore/exploit tradeoff, handled by Bayes instead of a tuning knob.

Context features, all in [0, 1]:
  gap_min:          1 - mastery of the problem's weakest topic
  gap_mean:         1 - mean mastery across the problem's topics
  difficulty_match: how well the problem's difficulty fits the level the
                    user's mastery implies (see target_difficulty). This
                    is the difficulty auto adjustment: cross a mastery
                    threshold and harder problems start scoring higher.
  recency:          days since the user last touched these topics, capped
                    at 30. Never touched counts as 30.
  bias:             constant 1, lets the model learn a global offset.

Reward: the change in mastery of the problem's weakest topic caused by
the attempt, clipped to [-1, 1]. Solving grows mastery (positive
reward), failing shrinks it (negative reward). The bandit learns which
contexts produce learning, not just which problems get solved.
"""

from __future__ import annotations

import numpy as np

FEATURES = ["gap_min", "gap_mean", "difficulty_match", "recency", "bias"]
D = len(FEATURES)

LAMBDA = 1.0    # prior precision: how skeptical we start
SIGMA2 = 0.1    # assumed reward noise variance
RECENCY_CAP_DAYS = 30.0

DIFFICULTY_NUM = {"Easy": 0.0, "Medium": 0.5, "Hard": 1.0}


def target_difficulty(mastery: float) -> float:
    """Difficulty level the user's mastery implies, on the 0/0.5/1 scale."""
    if mastery < 0.4:
        return 0.0
    if mastery < 0.7:
        return 0.5
    return 1.0


def weakest_topic(problem: dict, topic_mastery: dict) -> str:
    return min(problem["topics"], key=lambda t: topic_mastery.get(t, 0.5))


def context(problem: dict, state: dict, now: float) -> np.ndarray:
    from . import mastery as mst

    tm = {t: mst.mean(state, t, now) for t in problem["topics"]}
    ms = list(tm.values())
    gap_min = 1.0 - min(ms)
    gap_mean = 1.0 - sum(ms) / len(ms)
    weak_m = min(ms)
    dmatch = 1.0 - abs(DIFFICULTY_NUM[problem["difficulty"]] - target_difficulty(weak_m))
    lasts = [state.get("topics", {}).get(t, {}).get("last_ts") for t in problem["topics"]]
    if all(last is None for last in lasts):
        recency = 1.0
    else:
        days = min((now - last) / 86400.0 for last in lasts if last is not None)
        recency = min(max(days, 0.0), RECENCY_CAP_DAYS) / RECENCY_CAP_DAYS
    return np.array([gap_min, gap_mean, dmatch, recency, 1.0])


def get_posterior(state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, precision). Initializes a skeptical prior on first use."""
    b = state.setdefault("bandit", {})
    if "mean" not in b:
        b["mean"] = [0.0] * D
        b["prec"] = (np.eye(D) * LAMBDA).tolist()
    return np.array(b["mean"]), np.array(b["prec"])


def observe(state: dict, x: np.ndarray, reward: float) -> None:
    """Bayesian linear regression update for one (context, reward) pair."""
    mean, prec = get_posterior(state)
    prec_new = prec + np.outer(x, x) / SIGMA2
    rhs = prec @ mean + x * (reward / SIGMA2)
    mean_new = np.linalg.solve(prec_new, rhs)
    state["bandit"]["mean"] = mean_new.tolist()
    state["bandit"]["prec"] = prec_new.tolist()


def sample_theta(state: dict, seed: int | None = None) -> np.ndarray:
    mean, prec = get_posterior(state)
    cov = np.linalg.inv(prec)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov)


def score(theta: np.ndarray, x: np.ndarray) -> float:
    return float(theta @ x)


def explain(problem: dict, state: dict, now: float) -> list[str]:
    """Human readable reasons, from the posterior mean (deterministic)."""
    from . import mastery as mst

    x = context(problem, state, now)
    mean, _ = get_posterior(state)
    tm = {t: mst.mean(state, t, now) for t in problem["topics"]}
    weak = weakest_topic(problem, tm)
    lines = [f"weakest topic here: {weak} (mastery {tm[weak]:.0%})"]

    lasts = [state.get("topics", {}).get(t, {}).get("last_ts") for t in problem["topics"]]
    if all(last is None for last in lasts):
        lines.append("you have never practiced these topics")
    else:
        days = min((now - last) / 86400.0 for last in lasts if last is not None)
        lines.append(f"last touched {int(days)} days ago" if days >= 1 else "practiced today")

    driver = int(np.argmax(mean[:4] * x[:4]))
    names = {
        0: "attacking your biggest gap",
        1: "broad room to grow across these topics",
        2: f"difficulty {problem['difficulty']} fits your level",
        3: "spaced repetition: this topic is getting stale",
    }
    lines.append(names[driver])
    return lines
