"""Shadow skill model: ability-difficulty separation for next-attempt prediction.

The problem this asks about: every untouched topic defaults to 0.5
modeled mastery, and difficulty_match uses hard thresholds, while the
bank holds only 10 Hard problems. A failed Hard problem may reflect
difficulty as much as lack of topic competence, and the existing
topic means cannot tell those apart.

This module is a competing, separately maintained prediction head
for next-attempt correctness. It never writes state, never touches
``grindbandit.mastery``, the bandit posterior, or the reward. The
production prediction stays the existing per-topic means until a
human decides otherwise on real-data evidence. The registry at the
bottom of this module records that status for every candidate.

Observation
-----------
One observation is one attempt: ``problem_id``, ``difficulty``,
``topics``, ``solved``, and ``ts``. ``repeat`` marks an attempt on a
problem already attempted before, with ``prior_attempts_on_problem``
counting how many. One attempt yields one observation, however many
topics the problem carries. Attempts are user-selected (bandit picks
or manual choice), not randomly assigned problems, so no true ability
identification is claimed from one student and a sparse bank: these
are predictive checks, not causal estimates.

Candidates
----------
- ``global_rate``: the trivial baseline, the training solve rate for
  every holdout point.
- ``topic_mean``: the existing per-topic means, the production head.
  Each topic is a Beta(1, 1) posterior from training attempts only
  (one win or loss per attempt on each of the problem's topics); a
  problem's prediction averages its topics' posterior means and an
  unseen topic scores 0.5, the untouched-topic default.
- ``difficulty_heuristic``: the deterministic difficulty baseline,
  the training solve rate per difficulty, falling back to the global
  rate for a difficulty with no training attempts.
- ``hierarchical_beta``: a hierarchical Beta-binomial. Topic solve
  rates share a Beta prior centred on the global rate with a
  predeclared shrinkage strength of 8 pseudo-observations (the same
  pseudo-count GB14 uses for per-topic shrinkage), so a topic with
  little data stays close to the global rate and a topic with much
  data speaks for itself. The strength is fixed, not fitted: with
  multi-topic attempts, per-topic counts mix several true rates, so
  a count-based estimate of the shrinkage is not identified.
- ``rasch``: a small, regularized Rasch/1PL-style model. The logit
  of solving is a global ability plus a per-topic ability offset
  (shrunk toward zero with a tight prior) minus a per-difficulty
  offset, with Medium anchored at zero so ability and difficulty are
  identified. A Hard failure can load on the Hard offset instead of
  on topic ability. Fitted by penalized maximum likelihood (Newton
  steps on the penalized log-likelihood); every parameter is confined
  to [-4, 4] and a fit that lands on a bound, or fails to converge,
  is flagged as pathological and can never be adopted.

Bounds and pathological extremes
---------------------------------
A Rasch parameter on its bound, a fit that fails to converge, or a
training set with no outcome variation at all (every attempt solved,
or every attempt failed) is a pathological extreme: the data pushed
the model somewhere it cannot defend, or cannot identify anything
at all. Such a model can never be adopted.

Gate
----
The comparison uses a predeclared temporal holdout: the latest
quarter of observations by time. The gate needs at least 60 training
and 20 holdout observations, or the verdict is NOT IDENTIFIABLE. A
learned model is adopted only when it beats all three baselines on
holdout Brier score and log loss, and the paired bootstrap 95 percent
interval for its Brier improvement over the production model
(topic_mean, the head it would replace) lies entirely below zero,
and it carries no pathological flag. Anything else keeps the simple
model. The holdout is never used for fitting. The bootstrap is taken
against the production head, following the GB14 gate: the other two
baselines are reported as point comparisons, and the interval is the
robustness check on the decision that matters, replacing what is live.

Controlled simulation
---------------------
``simulate_observations`` builds attempts under a known ground
truth, in one of two families. The logit family is a global ability,
per-topic ability offsets, and per-difficulty offsets, combined as
``sigmoid(ability + mean topic offset - difficulty offset)``. The
rates family is per-topic solve rates and per-difficulty rates,
combined as the mean topic rate plus the difficulty rate's deviation
from 0.5, clipped to [0.02, 0.98]. Problems are sampled from the real
bank in both families. Simulation exists only to validate this
machinery against known answers. Simulated rows are labeled
``simulated`` and are never a substitute for real practice data;
``compare_models`` cannot tell simulated from real rows, so callers
must keep the two apart and label every verdict with its data source.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np

from grindbandit.research import calibration as cal

# --- Predeclared constants (set before any data is looked at) --------------

# Minimum observations for the gate to say anything at all. A rank
# or score comparison on fewer points is too noisy to promote a model
# on, so below these counts the honest verdict is NOT IDENTIFIABLE.
MIN_TRAIN_OBSERVATIONS = 60
MIN_HOLDOUT_OBSERVATIONS = 20

# Predeclared holdout share: the latest quarter of observations by
# time, set before any outcome is looked at.
HOLDOUT_FRACTION = 0.25

# Predeclared shrinkage strength for the hierarchical Beta-binomial,
# in pseudo-observations: a topic's rate is worth 8 global
# observations. The same pseudo-count GB14 uses for per-topic
# shrinkage. Fixed, not fitted: with multi-topic attempts, per-topic
# counts mix several true rates, so a count-based estimate of the
# strength is not identified (see the module docstring).
KAPPA = 8.0

# Bounds on any Rasch parameter (ability or difficulty offset), in
# logits. Outside [-4, 4] a parameter claims near-certainty, which
# sparse data cannot support.
RASCH_PARAM_BOUND = 4.0

# Rasch prior scales (standard deviations, in logits). Topic abilities
# are shrunk toward zero: a topic must earn its offset with evidence.
# Difficulty offsets and the global ability are shrunk more gently.
RASCH_SIGMA_ABILITY = 2.0
RASCH_SIGMA_TOPIC = 1.0
RASCH_SIGMA_DIFFICULTY = 1.5

# Newton fitting settings for the Rasch model.
RASCH_MAX_ITERATIONS = 100
RASCH_GRADIENT_TOLERANCE = 1e-6

# Paired bootstrap settings for the Brier improvement interval.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20270109

_SECONDS_PER_DAY = 86400.0
_SIMULATION_EPOCH = 1_700_000_000.0

_CANDIDATE_KINDS = ("global_rate", "topic_mean", "difficulty_heuristic",
                    "hierarchical_beta", "rasch")
_BASELINE_KINDS = ("global_rate", "topic_mean", "difficulty_heuristic")
_LEARNED_KINDS = ("hierarchical_beta", "rasch")
_DIFFICULTIES = ("Easy", "Medium", "Hard")

# --- Registry: which model is production, which are shadow ------------------
#
# Promotion is deliberately not a code path. A learned model moves out
# of "shadow" only by a human decision after this gate passes on real
# time-indexed outcomes, with a model flag and migration recorded at
# that time. Until then every learned candidate is shadow-only.

REGISTRY = {
    "topic_mean": {
        "version": "topic_mean/1.0",
        "status": "production",
        "description": "Existing per-topic Beta means, the production prediction head.",
    },
    "global_rate": {
        "version": "global_rate/1.0",
        "status": "baseline",
        "description": "Trivial baseline: the training solve rate for every point.",
    },
    "difficulty_heuristic": {
        "version": "difficulty_heuristic/1.0",
        "status": "baseline",
        "description": "Deterministic baseline: the training solve rate per difficulty.",
    },
    "hierarchical_beta": {
        "version": "hierarchical_beta/1.0",
        "status": "shadow",
        "description": "Hierarchical Beta-binomial: per-topic rates shrunk toward the global rate with a fixed strength of 8 pseudo-observations.",
    },
    "rasch": {
        "version": "rasch/1.0",
        "status": "shadow",
        "description": "Regularized Rasch-style model: per-topic ability minus per-difficulty offset.",
    },
}


def active_production_model() -> str:
    """The prediction head live mastery uses. Always ``topic_mean`` here."""
    return "topic_mean"


def registry_summary() -> dict:
    """A copy of the model registry: version and status per candidate."""
    return copy.deepcopy(REGISTRY)


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# --- Fitting -----------------------------------------------------------------


def _require_train(train: list[dict]) -> None:
    if not train:
        raise ValueError("cannot fit a model with no training observations")


def _topic_counts(train: list[dict]) -> dict[str, list[int]]:
    """Per-topic [solved, attempts] from training observations."""
    counts: dict[str, list[int]] = {}
    for ob in train:
        for topic in ob["topics"]:
            cell = counts.setdefault(topic, [0, 0])
            cell[0] += 1 if ob["solved"] else 0
            cell[1] += 1
    return counts


def _global_rate(train: list[dict]) -> float:
    return sum(1 for ob in train if ob["solved"]) / len(train)


def fit_global_rate(train: list[dict]) -> dict:
    _require_train(train)
    return {"kind": "global_rate", "rate": _global_rate(train),
            "pathological": []}


def fit_topic_mean(train: list[dict]) -> dict:
    """Per-topic Beta(1, 1) posterior means from training attempts."""
    _require_train(train)
    topic_means = {}
    for topic, (solved, attempts) in _topic_counts(train).items():
        topic_means[topic] = (1.0 + solved) / (2.0 + attempts)
    return {"kind": "topic_mean", "topic_means": topic_means,
            "pathological": []}


def fit_difficulty_heuristic(train: list[dict]) -> dict:
    """Training solve rate per difficulty, with the global rate as fallback."""
    _require_train(train)
    rates: dict[str, float] = {}
    for difficulty in _DIFFICULTIES:
        rows = [ob for ob in train if ob["difficulty"] == difficulty]
        if rows:
            rates[difficulty] = sum(1 for ob in rows if ob["solved"]) / len(rows)
    return {"kind": "difficulty_heuristic", "rates": rates,
            "global_rate": _global_rate(train), "pathological": []}


def fit_hierarchical_beta(train: list[dict]) -> dict:
    """Beta-binomial with fixed shrinkage toward the global rate.

    Each topic shares the prior Beta(KAPPA * mu, KAPPA * (1 - mu))
    centred on the training solve rate mu, so a topic's posterior
    mean is (KAPPA * mu + solved) / (KAPPA + attempts): a topic with
    one attempt stays near the global rate and a topic with many
    attempts speaks for itself. The strength is predeclared, not
    fitted from the counts: one attempt on a multi-topic problem
    contributes to several topics' counts at once, so the counts mix
    several true rates and cannot identify the shrinkage. A training
    set with no outcome variation (every attempt solved, or every
    attempt failed) collapses the prior onto 0 or 1 and is flagged
    degenerate.
    """
    _require_train(train)
    mu = _global_rate(train)
    counts = _topic_counts(train)
    pathological: list[str] = []
    if mu <= 0.0:
        pathological.append("degenerate_train_all_failed")
    elif mu >= 1.0:
        pathological.append("degenerate_train_all_solved")
    topic_means = {}
    for topic, (solved, attempts) in counts.items():
        topic_means[topic] = (KAPPA * mu + solved) / (KAPPA + attempts)
    return {"kind": "hierarchical_beta", "global_rate": mu, "kappa": KAPPA,
            "topic_means": topic_means, "pathological": pathological}


# --- Rasch-style model ---------------------------------------------------------
#
# logit p = ability + mean topic ability offset - difficulty offset,
# with Medium anchored at zero. Parameters are packed as
# [ability, u_topic..., b_Easy, b_Hard] with topics in sorted order.


def _rasch_topics(train: list[dict]) -> list[str]:
    return sorted({t for ob in train for t in ob["topics"]})


def _rasch_design(obs: list[dict], topics: list[str]) -> np.ndarray:
    """Design matrix for the Rasch parameters on ``obs``."""
    index = {t: i for i, t in enumerate(topics)}
    n_params = 1 + len(topics) + 2
    x = np.zeros((len(obs), n_params))
    for i, ob in enumerate(obs):
        x[i, 0] = 1.0
        for topic in ob["topics"]:
            j = index.get(topic)
            if j is not None:
                x[i, 1 + j] += 1.0 / len(ob["topics"])
        if ob["difficulty"] == "Easy":
            x[i, -2] = -1.0
        elif ob["difficulty"] == "Hard":
            x[i, -1] = -1.0
    return x


def _rasch_prior_precisions(n_topics: int) -> np.ndarray:
    prec = np.empty(1 + n_topics + 2)
    prec[0] = 1.0 / RASCH_SIGMA_ABILITY ** 2
    prec[1:1 + n_topics] = 1.0 / RASCH_SIGMA_TOPIC ** 2
    prec[-2:] = 1.0 / RASCH_SIGMA_DIFFICULTY ** 2
    return prec


def _rasch_objective_and_grad(params: np.ndarray, obs: list[dict],
                              topics: list[str]) -> tuple[float, np.ndarray]:
    """Penalized negative log-likelihood and its gradient at ``params``."""
    x = _rasch_design(obs, topics)
    y = np.array([1.0 if ob["solved"] else 0.0 for ob in obs])
    eta = x @ params
    # Stable logistic and log-likelihood.
    p = np.array([sigmoid(v) for v in eta])
    nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
    prec = _rasch_prior_precisions(len(topics))
    penalty = 0.5 * float(np.sum(prec * params ** 2))
    grad = x.T @ (p - y) + prec * params
    return nll + penalty, grad


def _rasch_pack(obs: list[dict], spec: dict) -> tuple[np.ndarray, list[str], tuple[str, ...]]:
    """Pack a spec dict into a Rasch parameter vector on ``obs`` topics."""
    topics = _rasch_topics(obs)
    params = np.zeros(1 + len(topics) + 2)
    params[0] = spec.get("ability", 0.0)
    topic_ability = spec.get("topic_ability", {})
    for j, topic in enumerate(topics):
        params[1 + j] = topic_ability.get(topic, 0.0)
    offsets = spec.get("difficulty_offset", {})
    params[-2] = offsets.get("Easy", 0.0)
    params[-1] = offsets.get("Hard", 0.0)
    return params, topics, ("Easy", "Hard")


def fit_rasch(train: list[dict]) -> dict:
    """Fit the Rasch-style model by penalized maximum likelihood.

    Newton steps on the penalized negative log-likelihood, with
    step-halving so the objective never increases. Parameters are
    clipped to [-4, 4] afterwards; a clipped parameter, or a fit that
    never converges, is pathological and can never be adopted.
    """
    _require_train(train)
    topics = _rasch_topics(train)
    x = _rasch_design(train, topics)
    y = np.array([1.0 if ob["solved"] else 0.0 for ob in train])
    prec = _rasch_prior_precisions(len(topics))
    params = np.zeros(1 + len(topics) + 2)
    value, grad = _rasch_objective_and_grad(params, train, topics)
    converged = False
    for _ in range(RASCH_MAX_ITERATIONS):
        if float(np.max(np.abs(grad))) < RASCH_GRADIENT_TOLERANCE:
            converged = True
            break
        eta = x @ params
        p = np.array([sigmoid(v) for v in eta])
        hess = x.T @ (x * (p * (1.0 - p))[:, None]) + np.diag(prec)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        scale = 1.0
        improved = False
        for _ in range(30):
            candidate = params - scale * step
            new_value, _ = _rasch_objective_and_grad(candidate, train, topics)
            if new_value < value:
                params, value = candidate, new_value
                improved = True
                break
            scale *= 0.5
        if not improved:
            break
        _, grad = _rasch_objective_and_grad(params, train, topics)
    else:
        converged = float(np.max(np.abs(grad))) < RASCH_GRADIENT_TOLERANCE
    pathological: list[str] = []
    if not converged:
        pathological.append("did_not_converge")
    clipped = np.clip(params, -RASCH_PARAM_BOUND, RASCH_PARAM_BOUND)
    if not np.array_equal(clipped, params):
        pathological.append("parameter_at_bound")
    params = clipped
    # No outcome variation in training means ability and difficulty
    # cannot be separated at all: the likelihood is monotone in the
    # ability, so the fit is driven by the prior and the bound, not
    # by evidence.
    if all(ob["solved"] for ob in train):
        pathological.append("degenerate_train_all_solved")
    elif not any(ob["solved"] for ob in train):
        pathological.append("degenerate_train_all_failed")
    topic_ability = {topic: float(params[1 + j]) for j, topic in enumerate(topics)}
    return {"kind": "rasch",
            "ability": float(params[0]),
            "topic_ability": topic_ability,
            "difficulty_offset": {"Easy": float(params[-2]), "Medium": 0.0,
                                  "Hard": float(params[-1])},
            "topics": topics,
            "neg_log_likelihood": float(value),
            "converged": bool(converged),
            "pathological": pathological}


def fit_model(kind: str, train: list[dict]) -> dict:
    """Fit one candidate on the training observations. Returns its params."""
    if kind == "global_rate":
        return fit_global_rate(train)
    if kind == "topic_mean":
        return fit_topic_mean(train)
    if kind == "difficulty_heuristic":
        return fit_difficulty_heuristic(train)
    if kind == "hierarchical_beta":
        return fit_hierarchical_beta(train)
    if kind == "rasch":
        return fit_rasch(train)
    raise ValueError(f"unknown skill model kind: {kind!r}")


# --- Prediction and scoring --------------------------------------------------


def _rasch_logit(model: dict, ob: dict) -> float:
    topic_offsets = [model["topic_ability"].get(t, 0.0) for t in ob["topics"]]
    mean_offset = sum(topic_offsets) / len(topic_offsets) if topic_offsets else 0.0
    return (model["ability"] + mean_offset
            - model["difficulty_offset"].get(ob["difficulty"], 0.0))


def predict(model: dict, obs: list[dict]) -> list[float]:
    """Predicted solve probability for each observation under ``model``."""
    kind = model["kind"]
    if kind == "global_rate":
        return [model["rate"]] * len(obs)
    if kind == "topic_mean":
        out = []
        for ob in obs:
            means = [model["topic_means"].get(t, 0.5) for t in ob["topics"]]
            out.append(sum(means) / len(means) if means else 0.5)
        return out
    if kind == "difficulty_heuristic":
        return [model["rates"].get(ob["difficulty"], model["global_rate"])
                for ob in obs]
    if kind == "hierarchical_beta":
        out = []
        for ob in obs:
            means = [model["topic_means"].get(t, model["global_rate"])
                     for t in ob["topics"]]
            out.append(sum(means) / len(means) if means else model["global_rate"])
        return out
    if kind == "rasch":
        return [sigmoid(_rasch_logit(model, ob)) for ob in obs]
    raise ValueError(f"unknown skill model kind: {kind!r}")


def split_holdout(obs: list[dict],
                  holdout_fraction: float = HOLDOUT_FRACTION) -> tuple[list[dict], list[dict]]:
    """Time-ordered train/holdout split of observations.

    Rows are ordered by the timestamp of the attempt and the latest
    ``holdout_fraction`` form the holdout. The split is declared
    before outcomes are examined, and the holdout is never used for
    fitting: it only checks whether a fit on earlier data still shows
    up later. A repeated attempt on a problem can straddle the split;
    that is the task itself (predict the next attempt given history),
    and the repeat is recorded on the observation.
    """
    ordered = sorted(obs, key=lambda o: o["ts"])
    n_holdout = int(len(ordered) * holdout_fraction)
    n_holdout = max(0, min(n_holdout, len(ordered)))
    if n_holdout == 0:
        return ordered, []
    return ordered[:-n_holdout], ordered[-n_holdout:]


def score_model(model: dict, holdout: list[dict]) -> dict:
    """Holdout scores for one fitted model, with calibration and breakdowns."""
    predictions = predict(model, holdout)
    outcomes = [ob["solved"] for ob in holdout]
    return {"kind": model["kind"],
            "n_holdout": len(holdout),
            "brier": cal.brier_score(predictions, outcomes),
            "log_loss": cal.log_loss(predictions, outcomes),
            "calibration_bins": cal.calibration_bins(predictions, outcomes),
            "breakdown_by_difficulty": cal.breakdown_by_difficulty(predictions, holdout),
            "breakdown_by_topic": cal.breakdown_by_topic(predictions, holdout)}


def compare_models(train: list[dict], holdout: list[dict]) -> dict:
    """Fit every candidate on train, score on holdout, and apply the gate.

    Verdicts: ``NOT_IDENTIFIABLE`` when the predeclared observation
    minimums are not met; ``ADOPT:<kind>`` when one learned model beats
    all three baselines on Brier and log loss with a bootstrap interval
    against the production model that excludes zero and no pathological
    flag; otherwise ``KEEP_SIMPLE`` with the failed conditions listed
    per model.
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

    best_baseline = min(_BASELINE_KINDS,
                        key=lambda k: report["models"][k]["brier"])
    production_preds = predict(fitted["topic_mean"], holdout)
    report["best_baseline"] = best_baseline
    for kind in _LEARNED_KINDS:
        entry = report["models"][kind]
        entry["brier_vs_production"] = cal.bootstrap_brier_difference(
            predict(fitted[kind], holdout), production_preds, outcomes,
            resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)

    passing = []
    failures: dict[str, list[str]] = {}
    for kind in _LEARNED_KINDS:
        entry = report["models"][kind]
        failed = []
        for base in _BASELINE_KINDS:
            base_entry = report["models"][base]
            if not entry["brier"] < base_entry["brier"]:
                failed.append(f"does not beat the {base} Brier score")
            if not entry["log_loss"] < base_entry["log_loss"]:
                failed.append(f"does not beat the {base} log loss")
        if not entry["brier_vs_production"]["ci_high"] < 0.0:
            failed.append("bootstrap interval for the Brier improvement "
                          "over the production topic means includes zero")
        if fitted[kind].get("pathological"):
            failed.append(
                "pathological fit: " + ", ".join(fitted[kind]["pathological"]))
        failures[kind] = failed
        if not failed:
            passing.append(kind)
    report["gate_failures"] = failures
    if passing:
        best = min(passing, key=lambda k: report["models"][k]["brier"])
        report["verdict"] = f"ADOPT:{best}"
        report["reason"] = (
            f"{best} beats all three baselines on holdout Brier and log loss "
            f"with a bootstrap interval excluding zero.")
    else:
        report["verdict"] = "KEEP_SIMPLE"
        report["reason"] = ("No learned model passed every gate condition; "
                            "the existing topic means stay.")
    return report


# --- Controlled simulation (validation only, never real data) ----------------


def simulate_observations(n: int, truth: dict, seed: int,
                          bank: list[dict] | None = None) -> list[dict]:
    """Build attempt observations under a known ground truth.

    ``truth`` selects one of two families. The logit family carries a
    global ``ability``, per-topic ability offsets in ``topic_ability``,
    and per-difficulty offsets in ``difficulty_offset``; each row's
    solve probability is ``sigmoid(ability + mean topic offset -
    difficulty offset)``. The rates family carries per-topic solve
    rates in ``topic_rate`` and per-difficulty rates in
    ``difficulty_rate``; each row's solve probability is the mean
    topic rate plus the difficulty rate's deviation from 0.5, clipped
    to [0.02, 0.98]. Anything missing counts as neutral (offset 0, or
    rate 0.5). Each row samples a problem from the bank with
    replacement and draws the outcome as a Bernoulli with that
    probability. Timestamps increase by half a day per observation so
    the temporal split is meaningful. Every row is labeled
    ``simulated``.
    """
    if bank is None:
        from grindbandit import problems

        bank = problems.load()
    rng = random.Random(seed)
    rates_family = "topic_rate" in truth or "difficulty_rate" in truth
    ability = truth.get("ability", 0.0)
    topic_ability = truth.get("topic_ability", {})
    difficulty_offset = truth.get("difficulty_offset", {})
    topic_rate = truth.get("topic_rate", {})
    difficulty_rate = truth.get("difficulty_rate", {})
    seen: dict[int, int] = {}
    rows = []
    for i in range(n):
        problem = bank[rng.randrange(len(bank))]
        if rates_family:
            mean_rate = sum(topic_rate.get(t, 0.5) for t in problem["topics"]) / len(problem["topics"])
            p = mean_rate + (difficulty_rate.get(problem["difficulty"], 0.5) - 0.5)
            p = min(max(p, 0.02), 0.98)
        else:
            offsets = [topic_ability.get(t, 0.0) for t in problem["topics"]]
            mean_offset = sum(offsets) / len(offsets) if offsets else 0.0
            p = sigmoid(ability + mean_offset
                        - difficulty_offset.get(problem["difficulty"], 0.0))
        solved = rng.random() < p
        ts = _SIMULATION_EPOCH + i * 0.5 * _SECONDS_PER_DAY
        prior = seen.get(problem["id"], 0)
        seen[problem["id"]] = prior + 1
        rows.append({
            "problem_id": problem["id"],
            "difficulty": problem["difficulty"],
            "topics": list(problem["topics"]),
            "solved": solved,
            "ts": ts,
            "attempt_index": i,
            "prior_attempts_on_problem": prior,
            "repeat": prior > 0,
            "simulated": True,
        })
    return rows
