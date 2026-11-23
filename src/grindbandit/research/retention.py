"""Shadow retention model: learnable half-life versus fixed decay.

The production mastery model decays with one fixed 30-day half-life.
Different topics, success histories, and retention intervals may imply
different forgetting curves, so this module asks, offline, whether a
learned half-life predicts later recall better than the fixed one.

Nothing here is production. This module reads observations and returns
predictions and a gate verdict; it never writes state, never touches
``grindbandit.mastery``, the bandit posterior, or the reward. The fixed
30-day decay stays the production model until a human decides otherwise
on real-data evidence. The registry at the bottom of this module records
that status for every candidate.

Observation
-----------
One observation is a later attempt on a topic after an earlier solved
attempt on the same topic: ``delta_days`` is the days between the two
attempts and ``solved`` is whether the later attempt was solved. The
earlier attempt must have been solved, because the model predicts
recall of learned material, not first-time success. ``prior_successes``
counts solved attempts on the topic before the later attempt. ``ts``
is the timestamp of the later attempt and ``last_ts`` the timestamp of
the earlier one. One attempt on a multi-topic problem yields one
observation per topic, so observations from a single attempt share an
outcome and are not independent; the holdout scores below are
predictive checks, not independent-sample inference.

Probability family
------------------
Predicted recall at elapsed time D with half-life h is
``p = 2 ** (-D / h)``. At D = h the model predicts a coin flip, at
D = 0 it predicts certain recall. Half-lives are fitted by maximum
Bernoulli likelihood on training periods only.

Candidates
----------
- ``global_rate``: the trivial baseline, the training solve rate for
  every holdout point.
- ``fixed30``: the production baseline, h fixed at 30 days.
- ``global_h``: one half-life learned from all training observations.
- ``topic_h``: a half-life per topic, shrunk in log space toward the
  global half-life with a pseudo-count of 8 observations, so a topic
  with little data stays close to the global value. A per-topic fit is
  attempted only with at least 3 observations on that topic.
- ``recurrence_h``: half-life grows monotonically with prior practice,
  ``log h = a + b * log1p(prior_successes)`` with ``b`` in [0, 1], so
  more practice can only lengthen, never shorten, the half-life.

Bounds and pathological extremes
---------------------------------
A fitted half-life is confined to [1, 365] days. A fit that lands on
either bound (for example, a train set with no failures at all, where
the likelihood keeps improving as h grows) is flagged ``at_bound``
and can never be adopted: it is a property of the data, not evidence
about forgetting.

Gate
----
The comparison uses a predeclared temporal holdout: the latest quarter
of observations by time. The gate needs at least 60 training and 20
holdout observations, or the verdict is NOT IDENTIFIABLE. A learned
model is adopted only when it beats both baselines on holdout Brier
score and log loss, and the paired bootstrap 95 percent interval for
its Brier improvement over fixed30 lies entirely below zero. Anything
else keeps the simple model. The holdout is never used for fitting.

Controlled simulation
---------------------
``simulate_observations`` builds observations under a known
ground-truth half-life. It exists only to validate this machinery
against known answers. Simulated rows are labeled ``simulated`` and
are never a substitute for real practice data; ``compare_models``
cannot tell simulated from real rows, so callers must keep the two
apart and label every verdict with its data source.
"""

from __future__ import annotations

import copy
import math
import random

from grindbandit.mastery import HALF_LIFE_DAYS as FIXED_HALF_LIFE_DAYS

# --- Predeclared constants (set before any data is looked at) --------------

# Bounds on any fitted half-life, in days. Outside this range a
# half-life stops describing forgetting and starts describing the
# data collection instead.
MIN_HALF_LIFE_DAYS = 1.0
MAX_HALF_LIFE_DAYS = 365.0

# Minimum observations for the gate to say anything at all. A rank
# or score comparison on fewer points is too noisy to promote a model
# on, so below these counts the honest verdict is NOT IDENTIFIABLE.
MIN_TRAIN_OBSERVATIONS = 60
MIN_HOLDOUT_OBSERVATIONS = 20

# Predeclared holdout share: the latest quarter of observations by
# time, set before any outcome is looked at.
HOLDOUT_FRACTION = 0.25

# Pseudo-count for log-space shrinkage of per-topic half-lives toward
# the global half-life. A topic with n observations gets weight n
# against weight 8 for the global value.
TOPIC_SHRINKAGE_PSEUDO_COUNT = 8.0

# Minimum observations on one topic before a per-topic fit is even
# attempted. Below this the topic uses the global half-life.
MIN_TOPIC_OBSERVATIONS = 3

# Paired bootstrap settings for the Brier improvement interval.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20270108

_SECONDS_PER_DAY = 86400.0
_SIMULATION_EPOCH = 1_700_000_000.0

_CANDIDATE_KINDS = ("global_rate", "fixed30", "global_h", "topic_h", "recurrence_h")
_LEARNED_KINDS = ("global_h", "topic_h", "recurrence_h")

# --- Registry: which model is production, which are shadow ------------------
#
# Promotion is deliberately not a code path. A learned model moves out
# of "shadow" only by a human decision after this gate passes on real
# time-indexed outcomes, with a model flag and migration recorded at
# that time. Until then every learned candidate is shadow-only.

REGISTRY = {
    "fixed30": {
        "version": "fixed30/1.0",
        "status": "production",
        "description": "Fixed 30-day half-life, the production mastery decay.",
    },
    "global_h": {
        "version": "global_h/1.0",
        "status": "shadow",
        "description": "One half-life learned from training observations.",
    },
    "topic_h": {
        "version": "topic_h/1.0",
        "status": "shadow",
        "description": "Per-topic half-lives shrunk toward the global value.",
    },
    "recurrence_h": {
        "version": "recurrence_h/1.0",
        "status": "shadow",
        "description": "Half-life growing monotonically with prior practice.",
    },
}


def active_production_model() -> str:
    """The retention model live mastery uses. Always ``fixed30`` here."""
    return "fixed30"


def registry_summary() -> dict:
    """A copy of the model registry: version and status per candidate."""
    return copy.deepcopy(REGISTRY)


# --- Probability family ------------------------------------------------------


def recall_probability(delta_days: float, half_life_days: float) -> float:
    """Predicted recall after ``delta_days`` with half-life ``half_life_days``.

    ``p = 2 ** (-D / h)``: certain recall at D = 0, a coin flip at
    D = h, a quarter at D = 2h.
    """
    if half_life_days <= 0:
        raise ValueError("half-life must be positive")
    if delta_days <= 0:
        return 1.0
    return 0.5 ** (delta_days / half_life_days)


def _clip_probability(p: float) -> float:
    return min(max(p, 1e-9), 1.0 - 1e-9)


def _neg_log_likelihood(half_lives: list[float], obs: list[dict]) -> float:
    """Bernoulli negative log-likelihood of observations under per-row half-lives."""
    total = 0.0
    for h, ob in zip(half_lives, obs):
        p = _clip_probability(recall_probability(ob["delta_days"], h))
        total += -math.log(p) if ob["solved"] else -math.log(1.0 - p)
    return total


def _golden_section_minimize(func, low: float, high: float,
                             tolerance: float = 1e-7) -> tuple[float, float]:
    """Deterministic golden-section search. Returns (argmin, min value)."""
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = func(c), func(d)
    while b - a > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = func(d)
    x = (a + b) / 2.0
    return x, func(x)


# --- Fitting -----------------------------------------------------------------


def fit_global_half_life(obs: list[dict]) -> dict:
    """Fit one half-life by maximum likelihood on ``obs``.

    Returns the fitted half-life in days, the training negative
    log-likelihood, and whether the fit landed on a bound (a
    pathological extreme that can never be adopted).
    """
    if not obs:
        raise ValueError("cannot fit a half-life with no observations")
    log_low = math.log(MIN_HALF_LIFE_DAYS)
    log_high = math.log(MAX_HALF_LIFE_DAYS)

    def nll(log_h: float) -> float:
        h = math.exp(log_h)
        return _neg_log_likelihood([h] * len(obs), obs)

    best_log_h, best_nll = _golden_section_minimize(nll, log_low, log_high)
    h = math.exp(best_log_h)
    # Log-scale tolerance: the search stops within 1e-7 of the bound,
    # which is a relative gap of about 1e-7 in h, far inside 1e-3.
    at_bound = best_log_h <= log_low + 1e-3 or best_log_h >= log_high - 1e-3
    return {"half_life": h, "neg_log_likelihood": best_nll, "at_bound": at_bound}


def fit_topic_half_lives(obs: list[dict], global_half_life: float) -> dict:
    """Fit per-topic half-lives, shrunk in log space toward the global one.

    A topic with at least ``MIN_TOPIC_OBSERVATIONS`` observations gets
    its own maximum-likelihood fit, then shrunk: ``log h_t`` is the
    observation-weighted mean of ``log h_mle_t`` and ``log h_global``
    with the shrinkage pseudo-count as the global weight. A topic
    with fewer observations uses the global half-life unchanged.
    Returns the shrunk half-life per topic plus the topics whose raw
    fit landed on a bound.
    """
    by_topic: dict[str, list[dict]] = {}
    for ob in obs:
        by_topic.setdefault(ob["topic"], []).append(ob)
    log_global = math.log(global_half_life)
    k = TOPIC_SHRINKAGE_PSEUDO_COUNT
    half_lives: dict[str, float] = {}
    at_bound_topics: list[str] = []
    for topic, rows in by_topic.items():
        if len(rows) < MIN_TOPIC_OBSERVATIONS:
            half_lives[topic] = global_half_life
            continue
        fit = fit_global_half_life(rows)
        if fit["at_bound"]:
            at_bound_topics.append(topic)
        log_shrunk = (len(rows) * math.log(fit["half_life"]) + k * log_global) / (len(rows) + k)
        half_lives[topic] = math.exp(log_shrunk)
    return {"half_lives": half_lives, "global_half_life": global_half_life,
            "at_bound_topics": at_bound_topics}


def fit_recurrence_model(obs: list[dict]) -> dict:
    """Fit ``log h = a + b * log1p(prior_successes)`` with ``b`` in [0, 1].

    The half-life can only grow with prior practice, never shrink:
    ``b`` is confined to non-negative values. ``a`` is fitted by
    golden-section search for each ``b`` on a fixed grid, and the best
    pair wins. Per-row half-lives are clamped into the sane bounds.
    """
    if not obs:
        raise ValueError("cannot fit a recurrence model with no observations")
    log_features = [math.log1p(ob.get("prior_successes", 0)) for ob in obs]

    def nll_for_b(b: float):
        def nll(a: float) -> float:
            half_lives = [
                min(max(math.exp(a + b * x), MIN_HALF_LIFE_DAYS), MAX_HALF_LIFE_DAYS)
                for x in log_features
            ]
            return _neg_log_likelihood(half_lives, obs)
        return nll

    best = None
    steps = 20
    for i in range(steps + 1):
        b = i / steps
        a, value = _golden_section_minimize(
            nll_for_b(b), math.log(MIN_HALF_LIFE_DAYS), math.log(MAX_HALF_LIFE_DAYS))
        if best is None or value < best[2]:
            best = (a, b, value)
    a, b, value = best
    at_bound = a <= math.log(MIN_HALF_LIFE_DAYS) + 1e-3
    return {"a": a, "b": b, "neg_log_likelihood": value, "at_bound": at_bound}


def fit_model(kind: str, train: list[dict]) -> dict:
    """Fit one candidate on the training observations. Returns its params."""
    if kind == "global_rate":
        rate = sum(1 for ob in train if ob["solved"]) / len(train) if train else 0.5
        return {"kind": kind, "rate": rate}
    if kind == "fixed30":
        return {"kind": kind, "half_life": FIXED_HALF_LIFE_DAYS}
    if kind == "global_h":
        fit = fit_global_half_life(train)
        return {"kind": kind, **fit}
    if kind == "topic_h":
        glob = fit_global_half_life(train)
        fit = fit_topic_half_lives(train, glob["half_life"])
        return {"kind": kind, **fit, "at_bound": glob["at_bound"]
                or bool(fit["at_bound_topics"])}
    if kind == "recurrence_h":
        fit = fit_recurrence_model(train)
        return {"kind": kind, **fit}
    raise ValueError(f"unknown retention model kind: {kind!r}")


# --- Prediction and scoring --------------------------------------------------


def _half_life_for(model: dict, ob: dict) -> float | None:
    kind = model["kind"]
    if kind in ("fixed30", "global_h"):
        return model["half_life"]
    if kind == "topic_h":
        return model["half_lives"].get(ob["topic"], model["global_half_life"])
    if kind == "recurrence_h":
        h = math.exp(model["a"] + model["b"] * math.log1p(ob.get("prior_successes", 0)))
        return min(max(h, MIN_HALF_LIFE_DAYS), MAX_HALF_LIFE_DAYS)
    return None


def predict(model: dict, obs: list[dict]) -> list[float]:
    """Predicted recall probability for each observation under ``model``."""
    if model["kind"] == "global_rate":
        return [model["rate"]] * len(obs)
    return [recall_probability(ob["delta_days"], _half_life_for(model, ob))
            for ob in obs]


def brier_score(predictions: list[float], outcomes: list[bool]) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    if not predictions:
        raise ValueError("cannot score no predictions")
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(predictions, outcomes)) / len(predictions)


def log_loss(predictions: list[float], outcomes: list[bool]) -> float:
    """Mean Bernoulli negative log-likelihood of the outcomes."""
    if not predictions:
        raise ValueError("cannot score no predictions")
    total = 0.0
    for p, y in zip(predictions, outcomes):
        p = _clip_probability(p)
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


def due_date_error(model: dict, obs: list[dict]) -> float | None:
    """Mean absolute error, in days, of the predicted due date.

    The due date is the day predicted recall falls to 0.5, which is
    the last practice time plus one half-life. The error is measured
    against the timestamp of the actual next attempt on that topic.
    Models with no half-life (the global rate) have no due date.
    """
    errors = []
    for ob in obs:
        h = _half_life_for(model, ob)
        if h is None:
            return None
        due_ts = ob["last_ts"] + h * _SECONDS_PER_DAY
        errors.append(abs(due_ts - ob["ts"]) / _SECONDS_PER_DAY)
    return sum(errors) / len(errors) if errors else None


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


# --- Holdout and gate --------------------------------------------------------


def split_holdout(obs: list[dict],
                  holdout_fraction: float = HOLDOUT_FRACTION) -> tuple[list[dict], list[dict]]:
    """Time-ordered train/holdout split of observations.

    Rows are ordered by the timestamp of the later attempt and the
    latest ``holdout_fraction`` form the holdout. The split is declared
    before outcomes are examined, and the holdout is never used for
    fitting: it only checks whether a fit on earlier data still shows
    up later.
    """
    ordered = sorted(obs, key=lambda o: o["ts"])
    n_holdout = int(len(ordered) * holdout_fraction)
    n_holdout = max(0, min(n_holdout, len(ordered)))
    if n_holdout == 0:
        return ordered, []
    return ordered[:-n_holdout], ordered[-n_holdout:]


def score_model(model: dict, holdout: list[dict]) -> dict:
    """Holdout scores for one fitted model, with calibration and due dates."""
    predictions = predict(model, holdout)
    outcomes = [ob["solved"] for ob in holdout]
    return {"kind": model["kind"],
            "n_holdout": len(holdout),
            "brier": brier_score(predictions, outcomes),
            "log_loss": log_loss(predictions, outcomes),
            "calibration_bins": calibration_bins(predictions, outcomes),
            "due_date_error_days": due_date_error(model, holdout)}


def compare_models(train: list[dict], holdout: list[dict]) -> dict:
    """Fit every candidate on train, score on holdout, and apply the gate.

    Verdicts: ``NOT_IDENTIFIABLE`` when the predeclared observation
    minimums are not met; ``ADOPT:<kind>`` when one learned model beats
    both baselines on Brier and log loss with a bootstrap interval
    that excludes zero and no fitted half-life on a bound; otherwise
    ``KEEP_SIMPLE`` with the failed conditions listed per model.
    """
    report: dict = {
        "n_train": len(train),
        "n_holdout": len(holdout),
        "min_train": MIN_TRAIN_OBSERVATIONS,
        "min_holdout": MIN_HOLDOUT_OBSERVATIONS,
        "models": {},
        "verdict": "NOT_IDENTIFIABLE",
        "reason": "",
    }
    if len(train) < MIN_TRAIN_OBSERVATIONS or len(holdout) < MIN_HOLDOUT_OBSERVATIONS:
        report["reason"] = (
            f"Need at least {MIN_TRAIN_OBSERVATIONS} training and "
            f"{MIN_HOLDOUT_OBSERVATIONS} holdout observations; "
            f"have {len(train)} and {len(holdout)}.")
        return report

    outcomes = [ob["solved"] for ob in holdout]
    fitted = {kind: fit_model(kind, train) for kind in _CANDIDATE_KINDS}
    for kind in _CANDIDATE_KINDS:
        entry = score_model(fitted[kind], holdout)
        entry.update({k: v for k, v in fitted[kind].items() if k != "kind"})
        report["models"][kind] = entry

    fixed_preds = predict(fitted["fixed30"], holdout)
    for kind in _LEARNED_KINDS:
        entry = report["models"][kind]
        entry["brier_vs_fixed30"] = bootstrap_brier_difference(
            predict(fitted[kind], holdout), fixed_preds, outcomes)

    base = report["models"]["global_rate"]
    fixed = report["models"]["fixed30"]
    passing = []
    failures: dict[str, list[str]] = {}
    for kind in _LEARNED_KINDS:
        entry = report["models"][kind]
        failed = []
        if not entry["brier"] < base["brier"]:
            failed.append("does not beat the global-rate Brier score")
        if not entry["brier"] < fixed["brier"]:
            failed.append("does not beat the fixed-30 Brier score")
        if not entry["log_loss"] < base["log_loss"]:
            failed.append("does not beat the global-rate log loss")
        if not entry["log_loss"] < fixed["log_loss"]:
            failed.append("does not beat the fixed-30 log loss")
        if not entry["brier_vs_fixed30"]["ci_high"] < 0.0:
            failed.append("bootstrap interval for the Brier improvement includes zero")
        if fitted[kind].get("at_bound"):
            failed.append("fitted half-life sits on a bound")
        failures[kind] = failed
        if not failed:
            passing.append(kind)
    report["gate_failures"] = failures
    if passing:
        best = min(passing, key=lambda k: report["models"][k]["brier"])
        report["verdict"] = f"ADOPT:{best}"
        report["reason"] = (
            f"{best} beats both baselines on holdout Brier and log loss "
            f"with a bootstrap interval excluding zero.")
    else:
        report["verdict"] = "KEEP_SIMPLE"
        report["reason"] = ("No learned model passed every gate condition; "
                            "the fixed 30-day model stays.")
    return report


# --- Controlled simulation (validation only, never real data) ----------------


def simulate_observations(n: int, half_lives, seed: int,
                          delta_min_days: float = 0.25,
                          delta_max_days: float = 30.0) -> list[dict]:
    """Build retention observations under a known ground-truth half-life.

    ``half_lives`` is one half-life in days for every topic, or a dict
    of topic to half-life. Elapsed times are log-uniform between
    ``delta_min_days`` and ``delta_max_days``; each outcome is a
    Bernoulli draw with probability ``2 ** (-D / h)`` for its topic.
    Timestamps increase by half a day per observation so the temporal
    split is meaningful. Every row is labeled ``simulated``.
    """
    rng = random.Random(seed)
    if isinstance(half_lives, dict):
        topics = sorted(half_lives)
        get_h = lambda t: half_lives[t]
    else:
        topics = ["array"]
        get_h = lambda t: float(half_lives)
    rows = []
    log_low, log_high = math.log(delta_min_days), math.log(delta_max_days)
    for i in range(n):
        topic = topics[i % len(topics)]
        delta = math.exp(rng.uniform(log_low, log_high))
        h = get_h(topic)
        solved = rng.random() < 0.5 ** (delta / h)
        ts = _SIMULATION_EPOCH + i * 0.5 * _SECONDS_PER_DAY
        rows.append({
            "topic": topic,
            "delta_days": delta,
            "solved": solved,
            "prior_successes": rng.randint(0, 6),
            "ts": ts,
            "last_ts": ts - delta * _SECONDS_PER_DAY,
            "simulated": True,
        })
    return rows
