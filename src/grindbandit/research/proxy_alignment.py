"""Proxy reward alignment laboratory.

The bandit optimizes a model-derived reward: the change in modeled
mastery of the weakest topic after an attempt. That reward can rise
mechanically even when delayed transfer on a different problem days
later does not move. This module checks, offline, whether ranking
actions by that reward predicts a better outcome on a delayed transfer
quiz.

Everything here reads state; nothing writes it, and nothing in this
module feeds mastery, the bandit posterior, or the reward. Quiz
outcomes are observational records and stay that way.

What the module does:

- Joins each attempt to its delayed transfer quiz by the immutable
  ``attempt_id``, preserving the realized delay, whether the quiz item
  was novel, how much practice exposure the quiz problem already had,
  and which outcomes are still missing.
- Splits linked observations into a training set and a predeclared,
  time-ordered holdout (the latest quarter by attempt time). The
  holdout is never used to tune anything; it only checks whether an
  association found on earlier data still shows up later.
- Computes rank association (Spearman) between the model-derived
  reward and the delayed outcome, discrimination (AUC) of the reward
  as a predictor of delayed success, calibration by reward tercile,
  and stratified failure maps (by delay timing, item novelty, practice
  exposure, and topic).
- Reports sample attrition and sensitivity bounds for missing or late
  outcomes instead of treating unobserved quizzes as failures.

Gate: the analysis is IDENTIFIABLE only with at least
``MIN_LINKED_OBSERVATIONS`` completed linked observations and at least
``MIN_HOLDOUT_OBSERVATIONS`` in the holdout. Below that the verdict is
NOT IDENTIFIABLE. Those thresholds are predeclared constants, not
fitted to any data.

Controlled simulation: ``simulate_linked_rows`` builds linked rows
under a known ground truth (aligned, where the delayed outcome really
does improve with the reward, or misaligned, where it does not). It
exists only to validate these diagnostics against known answers.
Simulated rows are labeled ``simulated`` and are never a substitute
for real practice data; ``analyze`` only ever reads a real state dict.

No causal claim is made anywhere here: a positive association would
show the proxy ranking tracks retention, not that the reward causes
it.
"""

from __future__ import annotations

import math
import random

# Predeclared gate: minimum completed linked observations (attempt with
# a recorded delayed quiz outcome) for an identifiable analysis, and
# minimum observations in the predeclared holdout. A rank association
# on fewer points is too noisy to say anything about alignment, so
# below these counts the honest verdict is NOT IDENTIFIABLE.
MIN_LINKED_OBSERVATIONS = 30
MIN_HOLDOUT_OBSERVATIONS = 10

# Predeclared holdout share: the latest quarter of linked observations
# by attempt time, set before any outcome is looked at.
HOLDOUT_FRACTION = 0.25

_SECONDS_PER_DAY = 86400.0
_SIMULATION_EPOCH = 1_700_000_000.0


# --- Rank statistics (pure Python, auditable by hand) -------------------


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based) with ties sharing the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation, or None when either side is constant."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0.0 or syy == 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, or None when it is undefined.

    Undefined for fewer than two points or when either side is
    constant (no ranks to compare).
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_ranks(list(xs)), _ranks(list(ys)))


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Area under the ROC curve (Mann-Whitney), or None.

    The probability that a randomly chosen delayed success scores
    above a randomly chosen delayed failure, with ties counting half.
    Undefined when either class is absent.
    """
    if len(scores) != len(labels):
        return None
    positives = [s for s, label in zip(scores, labels) if label]
    negatives = [s for s, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for q in negatives:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


# --- Join: attempts to delayed quizzes ----------------------------------


def _prior_attempts(log: list[dict], problem_id, topic: str | None,
                   before_ts: float) -> tuple[int, int]:
    """Attempts on a problem, and on a topic, before a timestamp."""
    problem_count = 0
    topic_count = 0
    for entry in log:
        ts = entry.get("ts")
        if ts is None or ts >= before_ts:
            continue
        if entry.get("problem_id") == problem_id:
            problem_count += 1
        if topic is not None and topic in (entry.get("topics") or []):
            topic_count += 1
    return problem_count, topic_count


def link_attempts_to_quizzes(state: dict) -> dict:
    """Join the attempt log to delayed quizzes by ``attempt_id``.

    Returns the linked rows plus attrition counts. A row preserves the
    realized delay (from the recorded outcome when there is one),
    the timing label, the assignment record (novel, prior assignments,
    pool exhausted), and practice exposure (attempts on the quiz
    problem and on its topic before the quiz was scheduled). Rows with
    no recorded outcome keep ``outcome_solved`` as None: missing is
    missing, never a failure. Attempts and quizzes that cannot be
    joined (an attempt with no id, a quiz naming no source attempt or
    an unknown one) are counted as attrition, not dropped silently.
    """
    log = state.get("log") or []
    quizzes = state.get("delayed_quizzes") or []
    attempts_by_id = {}
    n_attempts_without_id = 0
    for entry in log:
        aid = entry.get("attempt_id")
        if aid is None:
            n_attempts_without_id += 1
        else:
            attempts_by_id[aid] = entry
    rows = []
    n_quizzes_without_source = 0
    attempts_with_quiz = set()
    for quiz in quizzes:
        source_id = quiz.get("source_attempt_id")
        attempt = attempts_by_id.get(source_id) if source_id else None
        if attempt is None:
            n_quizzes_without_source += 1
            continue
        attempts_with_quiz.add(source_id)
        outcome = quiz.get("outcome") or {}
        scheduled_ts = quiz.get("scheduled_ts")
        delay_seconds = outcome.get("delay_seconds")
        if delay_seconds is not None:
            delay_days = delay_seconds / _SECONDS_PER_DAY
        elif quiz.get("delay_days") is not None:
            delay_days = float(quiz["delay_days"])
        else:
            delay_days = None
        assignment = quiz.get("assignment") or {}
        topic = quiz.get("topic")
        problem_prior, topic_prior = (0, 0)
        if scheduled_ts is not None:
            problem_prior, topic_prior = _prior_attempts(
                log, quiz.get("quiz_problem_id"), topic, scheduled_ts)
        rows.append({
            "attempt_id": source_id,
            "attempt_ts": attempt.get("ts"),
            "reward": attempt.get("reward"),
            "source_solved": attempt.get("solved"),
            "topic": topic,
            "delay_days": delay_days,
            "timing": outcome.get("timing"),
            "novel": assignment.get("novel"),
            "prior_assignments": assignment.get("prior_assignments", 0),
            "pool_exhausted": assignment.get("pool_exhausted", False),
            "quiz_problem_id": quiz.get("quiz_problem_id"),
            "quiz_problem_prior_attempts": problem_prior,
            "topic_prior_attempts": topic_prior,
            "outcome_solved": outcome.get("solved"),
            "outcome_status": quiz.get("status") or (
                "completed" if outcome else "pending"),
        })
    n_completed = sum(1 for r in rows if r["outcome_solved"] is not None)
    return {
        "rows": rows,
        "n_attempts": len(log),
        "n_attempts_without_id": n_attempts_without_id,
        "n_attempts_with_quiz": len(attempts_with_quiz),
        "n_attempts_without_quiz": len(log) - len(attempts_with_quiz),
        "n_quizzes": len(quizzes),
        "n_quizzes_without_source_attempt": n_quizzes_without_source,
        "n_linked": len(rows),
        "n_linked_completed": n_completed,
        "n_completed": n_completed,
        "n_pending": len(rows) - n_completed,
    }


# --- Holdout --------------------------------------------------------------


def split_holdout(rows: list[dict],
                  holdout_fraction: float = HOLDOUT_FRACTION) -> tuple[list[dict], list[dict]]:
    """Time-ordered train/holdout split of linked rows.

    Rows are ordered by attempt time and the latest
    ``holdout_fraction`` form the holdout. The split is declared
    before outcomes are examined, and the holdout is never used for
    tuning: it only checks whether an association holds up later.
    """
    ordered = sorted(rows, key=lambda r: (r["attempt_ts"] is None, r["attempt_ts"]))
    n_holdout = int(len(ordered) * holdout_fraction)
    n_holdout = max(0, min(n_holdout, len(ordered)))
    if n_holdout == 0:
        return ordered, []
    return ordered[:-n_holdout], ordered[-n_holdout:]


# --- Diagnostics on completed linked rows ---------------------------------


def _completed(rows: list[dict]) -> list[dict]:
    """Rows usable for association: a recorded outcome and a reward."""
    return [r for r in rows
            if r.get("outcome_solved") is not None
            and isinstance(r.get("reward"), (int, float))]


def calibration_by_tercile(rows: list[dict]) -> list[dict]:
    """Delayed solve rate by reward tercile (lowest to highest)."""
    done = _completed(rows)
    if not done:
        return []
    ordered = sorted(done, key=lambda r: r["reward"])
    terciles = []
    n = len(ordered)
    for i in range(3):
        chunk = ordered[i * n // 3:(i + 1) * n // 3]
        if not chunk:
            continue
        terciles.append({
            "tercile": i + 1,
            "n": len(chunk),
            "mean_reward": sum(r["reward"] for r in chunk) / len(chunk),
            "reward_low": chunk[0]["reward"],
            "reward_high": chunk[-1]["reward"],
            "delayed_solve_rate": sum(1 for r in chunk if r["outcome_solved"]) / len(chunk),
        })
    return terciles


def stratified_failure_map(rows: list[dict],
                           keys: tuple[str, ...] = ("timing", "novel", "topic")) -> list[dict]:
    """Delayed solve rate and mean reward per stratum of completed rows.

    Default strata are delay timing, item novelty, and topic; callers
    may pass any row keys (for example ``("novel", "timing")`` or
    ``("quiz_problem_prior_attempts",)`` for practice exposure). A
    stratum with a low solve rate despite high rewards is where the
    proxy is failing, not where practice is failing.
    """
    done = _completed(rows)
    groups: dict[tuple, list[dict]] = {}
    for r in done:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    strata = []
    for key_values, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        strata.append({
            **{k: v for k, v in zip(keys, key_values)},
            "n": len(members),
            "mean_reward": sum(r["reward"] for r in members) / len(members),
            "delayed_solve_rate": (
                sum(1 for r in members if r["outcome_solved"]) / len(members)),
        })
    return strata


def sensitivity_bounds(rows: list[dict]) -> dict:
    """Best/worst case bounds over missing outcomes.

    Missing outcomes are never treated as failures. The observed rate
    covers completed rows only; the worst case counts every missing
    outcome as failed and the best case as solved, and the Spearman
    association is recomputed under both imputations so the reader
    sees how much the missing data could move the answer.
    """
    done = _completed(rows)
    n_missing = sum(1 for r in rows if r.get("outcome_solved") is None)
    solved = sum(1 for r in done if r["outcome_solved"])
    total = len(rows)
    observed_rate = (solved / len(done)) if done else None
    worst_rate = (solved / total) if total else None
    best_rate = ((solved + n_missing) / total) if total else None
    observed = spearman([r["reward"] for r in done],
                        [1 if r["outcome_solved"] else 0 for r in done]) if done else None
    worst = best = None
    if total >= 2:
        rewards = [r["reward"] for r in rows
                   if isinstance(r.get("reward"), (int, float))]
        solved_flags = [r.get("outcome_solved") for r in rows
                        if isinstance(r.get("reward"), (int, float))]
        if len(rewards) == len(solved_flags) and len(rewards) >= 2:
            worst = spearman(rewards, [1 if s else 0 for s in solved_flags])
            best = spearman(rewards, [1 if (s is not False) else 0 for s in solved_flags])
    return {
        "n_completed": len(done),
        "n_missing": n_missing,
        "observed_rate": observed_rate,
        "worst_case_rate": worst_rate,
        "best_case_rate": best_rate,
        "spearman_observed": observed,
        "spearman_worst": worst,
        "spearman_best": best,
    }


def compute_diagnostics(rows: list[dict]) -> dict:
    """Rank association, discrimination, calibration, and strata."""
    done = _completed(rows)
    rewards = [r["reward"] for r in done]
    outcomes = [1 if r["outcome_solved"] else 0 for r in done]
    return {
        "n": len(done),
        "spearman": spearman(rewards, outcomes),
        "auc": auc(rewards, outcomes),
        "delayed_solve_rate": (sum(outcomes) / len(outcomes)) if outcomes else None,
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "calibration_by_tercile": calibration_by_tercile(rows),
        "stratified_failure_map": stratified_failure_map(rows),
        "sensitivity_bounds": sensitivity_bounds(rows),
    }


# --- Gate and full report ---------------------------------------------------


def diagnose(rows: list[dict],
             holdout_fraction: float = HOLDOUT_FRACTION) -> dict:
    """Verdict and diagnostics for a set of linked rows.

    IDENTIFIABLE only when there are at least
    ``MIN_LINKED_OBSERVATIONS`` completed linked observations and at
    least ``MIN_HOLDOUT_OBSERVATIONS`` in the predeclared holdout.
    Otherwise the verdict is NOT IDENTIFIABLE and no association is
    reported, because a rank statistic on a handful of points cannot
    say whether the proxy tracks retention.
    """
    done = _completed(rows)
    if len(done) < MIN_LINKED_OBSERVATIONS:
        return {
            "verdict": "NOT IDENTIFIABLE",
            "reason": (f"only {len(done)} completed linked observations; "
                       f"at least {MIN_LINKED_OBSERVATIONS} are needed "
                       "before a rank association can be trusted"),
            "n_linked_completed": len(done),
            "diagnostics": None,
        }
    train, holdout = split_holdout(done, holdout_fraction)
    if len(holdout) < MIN_HOLDOUT_OBSERVATIONS:
        return {
            "verdict": "NOT IDENTIFIABLE",
            "reason": (f"predeclared holdout holds only {len(holdout)} "
                       f"observations; at least {MIN_HOLDOUT_OBSERVATIONS} "
                       "are needed to check an association out of sample"),
            "n_linked_completed": len(done),
            "diagnostics": None,
        }
    diagnostics = compute_diagnostics(done)
    diagnostics["train"] = compute_diagnostics(train)
    diagnostics["holdout"] = compute_diagnostics(holdout)
    return {
        "verdict": "IDENTIFIABLE",
        "reason": (f"{len(done)} completed linked observations with "
                   f"{len(holdout)} in the predeclared holdout"),
        "n_linked_completed": len(done),
        "diagnostics": diagnostics,
    }


def analyze(state: dict,
            holdout_fraction: float = HOLDOUT_FRACTION) -> dict:
    """Run the laboratory on a state dict. Read-only: the state is untouched.

    Returns the verdict, the attrition counts from the join, and the
    diagnostics when the gate passes. With no real linked observations
    available the verdict is NOT IDENTIFIABLE; that is a finding about
    the data, not a failure of the diagnostics.
    """
    joined = link_attempts_to_quizzes(state)
    report = diagnose(joined["rows"], holdout_fraction)
    report["attrition"] = {k: v for k, v in joined.items() if k != "rows"}
    return report


# --- Controlled simulation (validation only, never real data) --------------


def simulate_linked_rows(n: int, alignment: str, seed: int) -> list[dict]:
    """Linked rows under a known ground truth, for validating diagnostics.

    ``alignment="aligned"``: the delayed outcome really does improve
    with the model-derived reward, so a working diagnostic must find a
    positive rank association. ``alignment="misaligned"``: the reward
    still varies, but the delayed outcome is independent of it, so a
    working diagnostic must find no association. Every row is labeled
    ``simulated``. This is a controlled check of the math, not data:
    simulated rows must never be fed to ``analyze`` as if they were
    real practice.
    """
    if alignment not in ("aligned", "misaligned"):
        raise ValueError(f"unknown alignment: {alignment!r}")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    topics = ["array", "dp", "graph"]
    rows = []
    for i in range(n):
        # Rewards spread across the full [-1, 1] range on purpose: the
        # simulator is a check of the diagnostics, so the reward
        # marginal is chosen to make an alignment signal detectable,
        # not to mimic the small mastery changes real attempts produce.
        reward = rng.uniform(-1.0, 1.0)
        if alignment == "aligned":
            p_solved = 0.15 + 0.70 * (reward + 1.0) / 2.0
        else:
            p_solved = 0.5
        rows.append({
            "attempt_id": f"att-sim-{i:04d}",
            "attempt_ts": _SIMULATION_EPOCH + i * _SECONDS_PER_DAY,
            "reward": reward,
            "source_solved": rng.random() < 0.6,
            "topic": topics[i % len(topics)],
            "delay_days": 3.0,
            "timing": "on_time",
            "novel": True,
            "prior_assignments": 0,
            "pool_exhausted": False,
            "quiz_problem_id": 1000 + i,
            "quiz_problem_prior_attempts": 0,
            "topic_prior_attempts": i,
            "outcome_solved": rng.random() < p_solved,
            "outcome_status": "completed",
            "simulated": True,
        })
    return rows
