"""Diagnostics-first off-policy evaluation feasibility gate (GB19).

Logged estimated propensities alone do not validate inverse
propensity scoring. Before any policy value is estimated, this
module checks whether the log can support the target policy at all:

- exact exposure linkage: every row must name the recommendation
  that served it, that recommendation must exist, and the logged
  action must be the one the recommendation selected;
- positivity/support: every action the target policy can take must
  have been served with a positive logging probability, and no taken
  action may carry a zero logging probability;
- propensity Monte Carlo error: the logged propensity is a Monte
  Carlo estimate (GB12). Its standard error is either recorded on
  the row or bounded from the recorded sample count by the worst
  case sqrt(p (1 - p) / samples) <= 0.5 / sqrt(samples). A row whose
  relative Monte Carlo error is too large fails, and a row with
  neither a recorded error nor a sample count cannot be checked and
  fails closed;
- importance weights: percentiles of pi/mu, the fraction above the
  reporting cap, and a refusal when any weight exceeds the hard cap;
- effective sample size: (sum w)^2 / sum w^2 as a fraction of n;
- time leakage: no outcome may be timestamped before its exposure,
  and no outcome may lack a timestamp;
- unique assessment coverage: every target action must have at
  least one observed outcome, and no row may carry a missing reward.

``diagnose`` returns every diagnostic plus an ``estimable`` flag and
a list of ``reasons``. When any reason is present the log does not
identify the target policy value and no estimator may run: the
estimators module refuses and returns no numeric value.

Thresholds are predeclared below, set before any data is looked at.
They are gates on identifiability, not tuning knobs: a log that
fails them yields "not estimable", never a fabricated score.
"""

from __future__ import annotations

import math

# Minimum rows for any estimate to be attempted. Below this the
# sample is too thin for even a diagnostic verdict to be stable.
MIN_ROWS = 30

# A logging propensity whose Monte Carlo standard error exceeds
# this fraction of the propensity itself is not known accurately
# enough to trust its weight.
MAX_MC_REL_ERROR = 0.10

# Importance weights above the reporting cap are counted; a weight
# above the hard cap refuses estimation outright.
WEIGHT_CAP = 20.0
MAX_WEIGHT = 50.0

# The effective sample size must be at least this fraction of n:
# below it a handful of rows dominate the estimate.
MIN_ESS_FRACTION = 0.10

# Percentiles reported for the importance-weight distribution.
WEIGHT_PERCENTILES = (50, 90, 95, 99)


def _percentile(sorted_values, pct):
    """Linear-interpolation percentile of an already sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    frac = rank - low
    return float(sorted_values[low] * (1.0 - frac)
                 + sorted_values[high] * frac)


def _mc_standard_error(row):
    """Monte Carlo standard error of a row's logged propensity.

    Uses the recorded per-row error when present; otherwise the
    worst-case binomial bound from the recorded sample count. Returns
    None when neither is recorded, which is uncheckable.
    """
    se = row.get("logging_probability_se")
    if se is not None:
        return float(se)
    samples = row.get("logging_probability_samples")
    if samples is not None and samples > 0:
        return 0.5 / math.sqrt(samples)
    return None


def diagnose(rows, target_actions, recommendations=None):
    """Run every feasibility diagnostic on logged rows.

    ``rows`` is a list of dicts with keys: ``attempt_id``,
    ``recommendation_id``, ``action``, ``logging_probability``,
    ``target_probability``, ``reward`` (None when missing),
    ``exposure_ts``, ``outcome_ts``, and optionally
    ``logging_probability_se`` or ``logging_probability_samples``.
    ``target_actions`` is the set of actions the target policy can
    take (target probability above zero). ``recommendations`` maps
    recommendation id to the logged recommendation record; without
    it exact exposure linkage cannot be verified and the gate fails
    closed.
    """
    rows = list(rows)
    target_actions = set(target_actions)
    reasons = []
    n = len(rows)

    # --- exact exposure linkage -------------------------------------
    linked = 0
    linkage_mismatches = 0
    if recommendations is None:
        reasons.append(
            "exact exposure linkage cannot be verified: no "
            "recommendation records supplied")
        linkage = {"checked": False, "linked": 0, "total": n,
                   "fraction": 0.0, "mismatches": 0}
    else:
        for row in rows:
            rec_id = row.get("recommendation_id")
            rec = recommendations.get(rec_id) if rec_id else None
            if rec is None:
                continue
            if rec.get("selected_problem_id") != row.get("action"):
                linkage_mismatches += 1
                continue
            linked += 1
        if linked < n:
            reasons.append(
                f"exposure linkage incomplete: {linked} of {n} rows "
                f"link to the recommendation that served them")
        if linkage_mismatches:
            reasons.append(
                f"exposure linkage mismatch: {linkage_mismatches} rows "
                f"log an action different from the one their "
                f"recommendation selected")
        linkage = {"checked": True, "linked": linked, "total": n,
                   "fraction": linked / n if n else 0.0,
                   "mismatches": linkage_mismatches}

    # --- missing outcomes -------------------------------------------
    missing_outcomes = sum(1 for row in rows
                           if row.get("reward") is None)
    if missing_outcomes:
        reasons.append(
            f"missing outcomes: {missing_outcomes} rows carry no "
            f"observed reward, so their contribution cannot be "
            f"weighted without inventing it")

    # --- positivity / support ---------------------------------------
    zero_support = []
    for action in sorted(target_actions, key=str):
        served = [row for row in rows
                  if row.get("action") == action
                  and (row.get("logging_probability") or 0.0) > 0.0]
        if not served:
            zero_support.append(action)
    support_fraction = (
        (len(target_actions) - len(zero_support)) / len(target_actions)
        if target_actions else 0.0)
    if zero_support:
        reasons.append(
            "zero support: target actions never served with positive "
            f"logging probability: {zero_support}")
    zero_propensity_rows = sum(
        1 for row in rows
        if (row.get("target_probability") or 0.0) > 0.0
        and (row.get("logging_probability") or 0.0) <= 0.0)
    if zero_propensity_rows:
        reasons.append(
            f"positivity violated: {zero_propensity_rows} rows were "
            f"taken under a zero logging propensity, so their "
            f"importance weight is undefined")
    support = {"fraction": support_fraction,
               "zero_support_actions": zero_support,
               "zero_propensity_rows": zero_propensity_rows}

    # --- propensity Monte Carlo error --------------------------------
    uncheckable = 0
    max_rel_error = 0.0
    mc_failures = 0
    for row in rows:
        mu = row.get("logging_probability") or 0.0
        if mu <= 0.0:
            continue
        se = _mc_standard_error(row)
        if se is None:
            uncheckable += 1
            continue
        rel = se / mu
        max_rel_error = max(max_rel_error, rel)
        if rel > MAX_MC_REL_ERROR:
            mc_failures += 1
    if uncheckable:
        reasons.append(
            f"propensity Monte Carlo error uncheckable on "
            f"{uncheckable} rows: neither a recorded standard error "
            f"nor a sample count is present")
    if mc_failures:
        reasons.append(
            f"propensity Monte Carlo error too large on {mc_failures} "
            f"rows: worst relative error {max_rel_error:.3f} exceeds "
            f"{MAX_MC_REL_ERROR}")
    mc_error = {"max_relative_error": max_rel_error,
                "rows_above_limit": mc_failures,
                "uncheckable_rows": uncheckable}

    # --- importance weights and effective sample size ----------------
    weights = []
    for row in rows:
        mu = row.get("logging_probability") or 0.0
        pi = row.get("target_probability") or 0.0
        if mu > 0.0:
            weights.append(pi / mu)
    weights_sorted = sorted(weights)
    weight_report = {
        f"p{p}": _percentile(weights_sorted, p)
        for p in WEIGHT_PERCENTILES
    }
    weight_report["max"] = max(weights) if weights else 0.0
    weight_report["fraction_above_cap"] = (
        sum(1 for w in weights if w > WEIGHT_CAP) / len(weights)
        if weights else 0.0)
    if weight_report["max"] > MAX_WEIGHT:
        reasons.append(
            f"importance weight {weight_report['max']:.1f} exceeds "
            f"the hard cap {MAX_WEIGHT}")
    sum_w = sum(weights)
    sum_w2 = sum(w * w for w in weights)
    ess = (sum_w * sum_w / sum_w2) if sum_w2 > 0.0 else 0.0
    ess_fraction = ess / n if n else 0.0
    if ess_fraction < MIN_ESS_FRACTION:
        reasons.append(
            f"effective sample size collapsed: ESS {ess:.1f} is "
            f"{ess_fraction:.3f} of n={n}, below the "
            f"{MIN_ESS_FRACTION} minimum")

    # --- time leakage -------------------------------------------------
    leakage = 0
    missing_outcome_ts = 0
    for row in rows:
        exposure = row.get("exposure_ts")
        outcome = row.get("outcome_ts")
        if row.get("reward") is None:
            continue
        if outcome is None:
            missing_outcome_ts += 1
        elif exposure is not None and outcome < exposure:
            leakage += 1
    if leakage:
        reasons.append(
            f"time leakage: {leakage} outcomes are timestamped "
            f"before their exposure")
    if missing_outcome_ts:
        reasons.append(
            f"time leakage: {missing_outcome_ts} observed outcomes "
            f"carry no outcome timestamp")
    time_leakage = {"count": leakage,
                    "missing_outcome_timestamps": missing_outcome_ts}

    # --- unique assessment coverage -----------------------------------
    observed_actions = {row.get("action") for row in rows
                        if row.get("reward") is not None}
    covered = observed_actions & target_actions
    coverage_fraction = (len(covered) / len(target_actions)
                         if target_actions else 0.0)
    if coverage_fraction < 1.0:
        reasons.append(
            "unique assessment coverage incomplete: only "
            f"{len(covered)} of {len(target_actions)} target actions "
            f"have an observed outcome")
    coverage = {"fraction": coverage_fraction,
                "observed_actions": sorted(observed_actions, key=str),
                "target_actions": sorted(target_actions, key=str)}

    if n < MIN_ROWS:
        reasons.append(
            f"too few rows: n={n} is below the {MIN_ROWS} minimum")

    return {
        "estimable": not reasons,
        "reasons": reasons,
        "n_rows": n,
        "linkage": linkage,
        "missing_outcomes": missing_outcomes,
        "support": support,
        "propensity_mc_error": mc_error,
        "importance_weights": weight_report,
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess_fraction,
        "time_leakage": time_leakage,
        "unique_assessment_coverage": coverage,
    }
