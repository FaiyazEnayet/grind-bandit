"""Gated off-policy estimators (GB19).

These estimators run only after
``grindbandit.research.ope_diagnostics.diagnose`` passes every
check. ``estimate`` refuses first and computes second: when the
diagnostics fail it returns no numeric policy value at all, so a
refusal can never be mistaken for a lift.

Estimators, all for the value of a target policy under logged data
from a logging policy with known propensities:

- IPS: mean of (pi/mu) * reward, with the sample standard error of
  the weighted terms.
- SNIPS (self-normalized IPS): sum of (pi/mu) * reward divided by
  the sum of (pi/mu), with a linearization (delta-method) standard
  error.
- Doubly robust: a per-action mean-reward direct model plus the IPS
  correction on its residuals. The direct model is the plain
  per-action empirical mean reward, which is adequate for the
  synthetic gate; it is not a production reward model and no claim
  is made that it transfers to real practice data.

Sensitivity: the IPS value recomputed with importance weights
trimmed at their 95th percentile, reported alongside the shift
from the untrimmed value, so a reader can see how much the estimate
depends on the largest weights.

The Monte Carlo uncertainty bound reported with each estimate is
the prespecified 3 standard errors used by the gate tests. The
estimators never see real practice data here: the scripted demo
history holds 30 attempts and no delayed quizzes, so nothing real
is identifiable, and these functions are validated on synthetic
logged data with a known expected reward instead.
"""

from __future__ import annotations

import math

from grindbandit.research import ope_diagnostics as od

# Prespecified Monte Carlo uncertainty bound, in estimator standard
# errors, used by the gate tests. Set before any estimate is looked
# at; it is not tuned to the data.
Z_BOUND = 3.0

# Trimming percentile for the sensitivity analysis.
TRIM_PERCENTILE = 95


def _mean_and_se(terms):
    """Mean and sample standard error of a list of terms."""
    n = len(terms)
    if n == 0:
        return 0.0, 0.0
    mean = sum(terms) / n
    if n == 1:
        return mean, 0.0
    var = sum((t - mean) ** 2 for t in terms) / (n - 1)
    return mean, math.sqrt(var / n)


def _percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    frac = rank - low
    return float(sorted_values[low] * (1.0 - frac)
                 + sorted_values[high] * frac)


def _weights_and_rewards(rows):
    weights = []
    rewards = []
    for row in rows:
        mu = row.get("logging_probability") or 0.0
        pi = row.get("target_probability") or 0.0
        weights.append(pi / mu if mu > 0.0 else 0.0)
        rewards.append(float(row.get("reward")))
    return weights, rewards


def ips(rows):
    """Inverse propensity scoring estimate and standard error."""
    weights, rewards = _weights_and_rewards(rows)
    terms = [w * r for w, r in zip(weights, rewards)]
    value, se = _mean_and_se(terms)
    return {"value": value, "standard_error": se,
            "mc_bound": Z_BOUND * se}


def snips(rows):
    """Self-normalized IPS estimate and linearization standard error."""
    weights, rewards = _weights_and_rewards(rows)
    n = len(weights)
    sum_w = sum(weights)
    if n == 0 or sum_w == 0.0:
        return {"value": 0.0, "standard_error": 0.0, "mc_bound": 0.0}
    value = sum(w * r for w, r in zip(weights, rewards)) / sum_w
    mean_w = sum_w / n
    residuals = [w * r - value * w
                 for w, r in zip(weights, rewards)]
    _, se_resid = _mean_and_se(residuals)
    se = se_resid / mean_w if mean_w else 0.0
    return {"value": value, "standard_error": se,
            "mc_bound": Z_BOUND * se}


def doubly_robust(rows, target_probabilities):
    """Doubly robust estimate with a per-action mean-reward model.

    The direct model is the per-action empirical mean reward. The
    value is the target-policy average of the model plus the mean
    IPS correction on the residuals, and the standard error is the
    sample standard error of the per-row correction terms.
    """
    totals = {}
    counts = {}
    for row in rows:
        action = row.get("action")
        totals[action] = totals.get(action, 0.0) + float(row["reward"])
        counts[action] = counts.get(action, 0) + 1
    model = {a: totals[a] / counts[a] for a in totals}
    direct = sum(pi * model.get(a, 0.0)
                 for a, pi in target_probabilities.items())
    weights, rewards = _weights_and_rewards(rows)
    correction = []
    for row, w, r in zip(rows, weights, rewards):
        model_value = model.get(row.get("action"), 0.0)
        correction.append(w * (r - model_value))
    mean_correction, se = _mean_and_se(correction)
    value = direct + mean_correction
    return {"value": value, "standard_error": se,
            "mc_bound": Z_BOUND * se,
            "direct_model_value": direct}


def _trimmed_ips(rows):
    """IPS with weights trimmed at the 95th percentile, and the shift."""
    weights, rewards = _weights_and_rewards(rows)
    cap = _percentile(sorted(weights), TRIM_PERCENTILE)
    trimmed = [min(w, cap) * r for w, r in zip(weights, rewards)]
    value, _ = _mean_and_se(trimmed)
    base = ips(rows)["value"]
    return {"value": value, "trim_cap": cap,
            "shift_from_untrimmed": value - base}


def estimate(rows, target_probabilities, target_actions=None,
             recommendations=None):
    """Gated policy-value estimate.

    Runs the diagnostics first. When they fail, returns
    ``estimable: False`` with the reasons and no numeric value for
    any estimator. When they pass, returns IPS, SNIPS, and doubly
    robust values, each with a standard error and the prespecified
    Monte Carlo bound, plus a weight-trimming sensitivity check.
    """
    rows = list(rows)
    if target_actions is None:
        target_actions = set(target_probabilities)
    diagnostics = od.diagnose(rows, target_actions,
                              recommendations=recommendations)
    if not diagnostics["estimable"]:
        return {"estimable": False,
                "reasons": diagnostics["reasons"],
                "diagnostics": diagnostics,
                "ips": None, "snips": None, "doubly_robust": None,
                "sensitivity": None}
    return {"estimable": True,
            "reasons": [],
            "diagnostics": diagnostics,
            "ips": ips(rows),
            "snips": snips(rows),
            "doubly_robust": doubly_robust(rows, target_probabilities),
            "sensitivity": _trimmed_ips(rows)}
