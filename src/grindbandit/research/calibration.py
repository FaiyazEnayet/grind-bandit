"""Scoring for next-attempt correctness predictions.

Small, dependency-free helpers shared by the shadow skill model:
Brier score, log loss, calibration bins, a paired bootstrap interval
for a Brier improvement, and per-topic and by-difficulty breakdowns
of a holdout. A breakdown counts one observation under each of its
topics, so per-topic rows share outcomes and are predictive checks,
not independent-sample inference.
"""

from __future__ import annotations

import math
import random

# Paired bootstrap settings for Brier improvement intervals.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20270109


def clip_probability(p: float, eps: float = 1e-9) -> float:
    """Keep a predicted probability off exactly 0 and 1, where log
    loss is undefined. The clip is far inside any real difference."""
    return min(max(p, eps), 1.0 - eps)


def brier_score(predictions: list[float], outcomes: list[bool]) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    if not predictions:
        raise ValueError("cannot score no predictions")
    return sum((p - (1.0 if y else 0.0)) ** 2
               for p, y in zip(predictions, outcomes)) / len(predictions)


def log_loss(predictions: list[float], outcomes: list[bool]) -> float:
    """Mean Bernoulli negative log-likelihood of the outcomes."""
    if not predictions:
        raise ValueError("cannot score no predictions")
    total = 0.0
    for p, y in zip(predictions, outcomes):
        p = clip_probability(p)
        total += -math.log(p) if y else -math.log(1.0 - p)
    return total / len(predictions)


def calibration_bins(predictions: list[float], outcomes: list[bool],
                     edges: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                 0.6, 0.7, 0.8, 0.9, 1.0)) -> list[dict]:
    """Per-bin counts, mean predicted probability, and observed solve rate."""
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        idx = [i for i, p in enumerate(predictions)
               if (low <= p < high) or (high == 1.0 and p == 1.0)]
        if not idx:
            continue
        mean_pred = sum(predictions[i] for i in idx) / len(idx)
        observed = sum(1 for i in idx if outcomes[i]) / len(idx)
        bins.append({"bin_low": low, "bin_high": high, "n": len(idx),
                     "mean_predicted": mean_pred, "observed_rate": observed})
    return bins


def bootstrap_brier_difference(predictions_a: list[float],
                               predictions_b: list[float],
                               outcomes: list[bool],
                               resamples: int = BOOTSTRAP_RESAMPLES,
                               seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired bootstrap interval for mean Brier(a) minus mean Brier(b).

    Resampling is over holdout observations, paired so the same
    resampled rows score both models. Returns the observed difference
    and the percentile 95 percent interval.
    """
    n = len(outcomes)
    if n == 0:
        raise ValueError("cannot bootstrap no observations")
    rng = random.Random(seed)
    diffs_a = [(pa - (1.0 if y else 0.0)) ** 2 for pa, y in zip(predictions_a, outcomes)]
    diffs_b = [(pb - (1.0 if y else 0.0)) ** 2 for pb, y in zip(predictions_b, outcomes)]
    observed = sum(diffs_a) / n - sum(diffs_b) / n
    boot = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(sum(diffs_a[i] for i in idx) / n - sum(diffs_b[i] for i in idx) / n)
    boot.sort()
    low = boot[int(0.025 * resamples)]
    high = boot[min(int(0.975 * resamples), resamples - 1)]
    return {"mean_diff": observed, "ci_low": low, "ci_high": high}


def _breakdown_row(label: str, indices: list[int],
                   predictions: list[float], outcomes: list[bool]) -> dict:
    n = len(indices)
    mean_pred = sum(predictions[i] for i in indices) / n
    observed = sum(1 for i in indices if outcomes[i]) / n
    brier = sum((predictions[i] - (1.0 if outcomes[i] else 0.0)) ** 2
                for i in indices) / n
    ll = sum(-math.log(clip_probability(predictions[i])) if outcomes[i]
             else -math.log(1.0 - clip_probability(predictions[i]))
             for i in indices) / n
    return {"n": n, "mean_predicted": mean_pred, "observed_rate": observed,
            "brier": brier, "log_loss": ll}


def breakdown_by_difficulty(predictions: list[float], obs: list[dict]) -> list[dict]:
    """Holdout scores grouped by problem difficulty, in bank order."""
    outcomes = [ob["solved"] for ob in obs]
    rows = []
    for difficulty in ("Easy", "Medium", "Hard"):
        idx = [i for i, ob in enumerate(obs) if ob["difficulty"] == difficulty]
        if not idx:
            continue
        row = _breakdown_row(difficulty, idx, predictions, outcomes)
        row = {"difficulty": difficulty, **row}
        rows.append(row)
    return rows


def breakdown_by_topic(predictions: list[float], obs: list[dict]) -> list[dict]:
    """Holdout scores grouped by topic. One observation is counted
    under each of its topics, so rows share outcomes."""
    outcomes = [ob["solved"] for ob in obs]
    topics = sorted({t for ob in obs for t in ob["topics"]})
    rows = []
    for topic in topics:
        idx = [i for i, ob in enumerate(obs) if topic in ob["topics"]]
        row = _breakdown_row(topic, idx, predictions, outcomes)
        row = {"topic": topic, **row}
        rows.append(row)
    return rows
