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
contexts produce learning, not just which problems get solved. Note
the reward is model-derived: it is computed by the same mastery model
the selector uses, so by itself it is not independent evidence that a
recommendation caused learning. Every recommendation is therefore
logged in full (see build_recommendation) so picks can be audited and
reconstructed, and delayed transfer quiz outcomes are stored
separately as observational records that never feed this reward.
"""

from __future__ import annotations

import uuid

import numpy as np

FEATURES = ["gap_min", "gap_mean", "difficulty_match", "recency", "bias"]
D = len(FEATURES)

# Version of the context/feature layout produced by context(). Bump this
# whenever FEATURES or context() changes, so recommendation records logged
# under older layouts stay interpretable; a record replays only under the
# version that produced it.
FEATURE_VERSION = "context-v1"

# Monte Carlo draws used to estimate per-candidate selection probabilities.
PROBABILITY_SAMPLES = 10_000

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


def estimate_selection_probabilities(
    state: dict,
    contexts: list[np.ndarray],
    seed: int,
    samples: int = PROBABILITY_SAMPLES,
) -> list[float]:
    """Monte Carlo estimate of each candidate's probability of being picked.

    Draws `samples` thetas from the current posterior on a private RNG
    stream seeded by `seed`, so estimating never perturbs the stream used
    for the actual Thompson draw. The estimate is the win frequency with
    add-one (Laplace) smoothing, so every candidate gets a value in (0, 1)
    and the values sum to 1. Candidates must be ordered by problem id so
    ties resolve the same way as the actual pick (lowest id wins). These
    are probabilities under the model posterior, not measured frequencies
    and not evidence about learning.
    """
    mean, prec = get_posterior(state)
    cov = np.linalg.inv(prec)
    rng = np.random.default_rng(seed)
    thetas = rng.multivariate_normal(mean, cov, size=samples)
    X = np.asarray(contexts, dtype=float)
    winners = np.argmax(thetas @ X.T, axis=1)
    counts = np.bincount(winners, minlength=len(contexts)).astype(float)
    return ((counts + 1.0) / (samples + len(contexts))).tolist()


def build_recommendation(
    state: dict,
    candidates: list[dict],
    now: float,
    seed: int,
    review_mode: bool = False,
    samples: int = PROBABILITY_SAMPLES,
) -> dict:
    """Make a Thompson pick and return a self-contained log record for it.

    The record carries everything needed to reconstruct the pick later:
    the feature version, the RNG seed of the actual draw, the posterior
    snapshot, the per-topic snapshot, and every candidate's context
    vector. Selection probabilities are Monte Carlo estimates from the
    posterior, computed on a separate RNG stream so that estimating them
    does not change the draw itself.
    """
    from . import mastery as mst

    ordered = sorted(candidates, key=lambda p: p["id"])
    contexts = [context(p, state, now) for p in ordered]
    estimate_seed = (int(seed) + 1) % (2**63)
    probs = estimate_selection_probabilities(
        state, contexts, seed=estimate_seed, samples=samples
    )
    theta = sample_theta(state, seed=seed)
    scores = [score(theta, x) for x in contexts]
    winner = min(range(len(ordered)), key=lambda i: (-scores[i], ordered[i]["id"]))
    mean, prec = get_posterior(state)

    topic_names = sorted({t for p in ordered for t in p["topics"]})
    topics_snapshot = {}
    for t in topic_names:
        info = state.get("topics", {}).get(t, {})
        topics_snapshot[t] = {
            "alpha": info.get("alpha", 1.0),
            "beta": info.get("beta", 1.0),
            "last_ts": info.get("last_ts"),
            "mastery": mst.mean(state, t, now),
        }

    return {
        "id": f"rec-{uuid.uuid4().hex[:12]}",
        "ts": now,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURES),
        "seed": int(seed),
        "review_mode": bool(review_mode),
        "posterior": {"mean": mean.tolist(), "prec": prec.tolist()},
        "topics": topics_snapshot,
        "probability_estimate": {
            "method": "monte_carlo_posterior_resampling_add_one_smoothed",
            "samples": samples,
            "seed": estimate_seed,
            "note": "Model estimate of how likely each candidate was to be "
                    "picked under the current posterior. Not a measured "
                    "frequency and not evidence of learning benefit.",
        },
        "candidates": [
            {
                "problem_id": p["id"],
                "title": p["title"],
                "difficulty": p["difficulty"],
                "topics": list(p["topics"]),
                "context": [float(v) for v in x],
                "estimated_selection_probability": probs[i],
                "selected": i == winner,
            }
            for i, (p, x) in enumerate(zip(ordered, contexts))
        ],
        "selected_problem_id": ordered[winner]["id"],
    }


def replay_selection(record: dict) -> int:
    """Reconstruct the selected problem id from a logged recommendation.

    Uses only the record: the posterior snapshot, the candidate context
    vectors, and the recorded seed. Mirrors the selection rule in
    build_recommendation (highest sampled score, ties to the lowest id).
    """
    if record.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"record was logged under {record.get('feature_version')!r}, "
            f"this code replays {FEATURE_VERSION!r}"
        )
    snapshot = {"bandit": {"mean": record["posterior"]["mean"],
                           "prec": record["posterior"]["prec"]}}
    theta = sample_theta(snapshot, seed=record["seed"])
    best_key = None
    best_id = None
    for cand in record["candidates"]:
        s = score(theta, np.asarray(cand["context"], dtype=float))
        key = (-s, cand["problem_id"])
        if best_key is None or key < best_key:
            best_key = key
            best_id = cand["problem_id"]
    return best_id


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
