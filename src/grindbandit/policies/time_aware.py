"""Budget-aware candidate features and policies (policy-v1, shadow only).

The question: does recommending the highest predicted gain per pick
waste learner time versus maximizing transfer improvement per minute
or deadline-weighted retention? These features and policies are the
apparatus for asking it. Nothing here runs in live selection: the
registry records every challenger as shadow, and the production
selector is untouched.

Candidate features, all versioned under policy-v1:
  budget_minutes:   the user-stated remaining time for the session.
  expected_minutes: historically observed minutes for the problem's
                    difficulty, leave-one-problem-out: attempts on the
                    problem itself never predict its own duration, so
                    a problem cannot leak its own recorded time into
                    its own expectation. Falls back to a predeclared
                    difficulty prior when a difficulty has no history.
  difficulty:       0 / 0.5 / 1 for Easy / Medium / Hard.
  novelty:          1 when the problem was never attempted, else 0.
  interval_days:    days since the problem's weakest topic was last
                    practiced, capped at 30; a never-practiced topic
                    counts as the full cap.

Utility: U = predicted delayed-transfer benefit minus lambda times
expected time. The predicted benefit is the sampled LinTS score in
model-derived reward units, which is explicitly hypothetical as a
transfer predictor until a valid predictive head exists: the score
is a proxy for mastery change, not a measurement of delayed
transfer, and no policy here is adopted on its strength alone.
"""

from __future__ import annotations

from grindbandit import bandit, mastery

from .registry import POLICY_VERSION  # noqa: F401  (re-exported)

# Predeclared difficulty priors for expected minutes, in minutes.
# Starting heuristics, not fitted values: with no history on a
# difficulty there is nothing to fit from, so the prior is the only
# honest expectation. A short problem is 15 minutes, a medium one
# 30, a hard one 45.
DIFFICULTY_PRIOR_MINUTES = {"Easy": 15.0, "Medium": 30.0, "Hard": 45.0}

# Reward units charged per expected minute in the cost-aware
# utility. A sampled LinTS score lives roughly in [-1, 1] reward
# units, so 0.01 per minute prices a 30 minute problem at 0.3
# reward units: enough to break ties toward shorter problems
# without letting duration dominate the predicted benefit.
LAMBDA_COST = 0.01

# Longest interval a feature can report, in days. Matches the
# recency cap in the bandit context, so staleness means the same
# thing in both places.
INTERVAL_CAP_DAYS = 30.0


def expected_minutes(problem: dict, attempts: list[dict]) -> float:
    """Expected minutes for a problem from past attempts.

    The mean of recorded minutes on other problems of the same
    difficulty, excluding attempts on this problem itself: a
    problem's own recorded minutes never predict its own duration,
    so there is no naive target leakage. With no other attempt on
    the difficulty, the predeclared difficulty prior applies.
    """
    difficulty = problem["difficulty"]
    others = [a["minutes"] for a in attempts
              if a.get("difficulty") == difficulty
              and a.get("problem_id") != problem["id"]
              and a.get("minutes") is not None]
    if others:
        return sum(others) / len(others)
    return DIFFICULTY_PRIOR_MINUTES[difficulty]


def novelty(problem: dict, state: dict) -> float:
    """1.0 when the problem was never attempted, else 0.0."""
    pid = problem["id"]
    attempted = any(e.get("problem_id") == pid
                    for e in state.get("log") or [])
    record = (state.get("problems") or {}).get(str(pid), {})
    if record.get("last_ts") is not None:
        attempted = True
    return 0.0 if attempted else 1.0


def interval_days(problem: dict, state: dict, now: float) -> float:
    """Days since the problem's weakest topic was last practiced.

    The weakest topic is the one with the lowest modeled mastery.
    A never-practiced topic counts as the full 30-day cap: an
    untouched topic is the stalest signal a problem can carry.
    """
    tm = {t: mastery.mean(state, t, now) for t in problem["topics"]}
    weak = bandit.weakest_topic(problem, tm)
    last = state.get("topics", {}).get(weak, {}).get("last_ts")
    if last is None:
        return INTERVAL_CAP_DAYS
    days = (now - last) / 86400.0
    return min(max(days, 0.0), INTERVAL_CAP_DAYS)


def candidate_features(problem: dict, state: dict, now: float,
                       budget_minutes: float) -> dict:
    """One versioned feature bundle for a candidate problem."""
    expected = expected_minutes(problem, state.get("log") or [])
    return {
        "policy_version": POLICY_VERSION,
        "budget_minutes": float(budget_minutes),
        "expected_minutes": float(expected),
        "difficulty": bandit.DIFFICULTY_NUM[problem["difficulty"]],
        "novelty": novelty(problem, state),
        "interval_days": interval_days(problem, state, now),
        "remaining_minutes": float(budget_minutes) - float(expected),
    }


def utility(predicted_benefit: float, expected_minutes_value: float,
            lam: float = LAMBDA_COST) -> float:
    """U = predicted delayed-transfer benefit minus lambda times
    expected time. Hypothetical until a valid predictive head
    exists: the benefit is a model-derived score, not measured
    transfer."""
    return float(predicted_benefit) - float(lam) * float(expected_minutes_value)


def _gap_min(problem: dict, state: dict, now: float) -> float:
    tm = {t: mastery.mean(state, t, now) for t in problem["topics"]}
    return 1.0 - min(tm.values())


def _is_due_review(problem: dict, state: dict, now: float) -> bool:
    """Whether the problem is a due spaced review under the same
    rule live selection uses: solved before, and the review
    interval has elapsed since the last attempt."""
    record = (state.get("problems") or {}).get(str(problem["id"]), {})
    if not record.get("ever_solved"):
        return False
    last = record.get("last_ts")
    if last is None:
        return True
    age_days = (now - last) / 86400.0
    return age_days >= bandit.review_interval_days(state)


def _pick_lints(candidates, state, now, budget_minutes, seed, samples):
    record = bandit.build_recommendation(state, candidates, now,
                                         seed=seed, samples=samples)
    wanted = record["selected_problem_id"]
    return next(p for p in candidates if p["id"] == wanted)


def _pick_shortest_duration(candidates, state, now, budget_minutes,
                            seed, samples):
    attempts = state.get("log") or []
    return min(candidates,
               key=lambda p: (expected_minutes(p, attempts), p["id"]))


def _pick_weakest_topic(candidates, state, now, budget_minutes,
                        seed, samples):
    return min(candidates,
               key=lambda p: (-_gap_min(p, state, now), p["id"]))


def _pick_due_first(candidates, state, now, budget_minutes,
                    seed, samples):
    return min(candidates,
               key=lambda p: (0 if _is_due_review(p, state, now) else 1,
                              -interval_days(p, state, now), p["id"]))


def _pick_cost_aware_lints(candidates, state, now, budget_minutes,
                           seed, samples):
    theta = bandit.sample_theta(state, seed=seed)
    attempts = state.get("log") or []
    return min(
        candidates,
        key=lambda p: (
            -utility(bandit.score(theta, bandit.context(p, state, now)),
                     expected_minutes(p, attempts)),
            p["id"],
        ),
    )


def _pick_constrained_lints(candidates, state, now, budget_minutes,
                            seed, samples):
    attempts = state.get("log") or []
    fitting = [p for p in candidates
               if expected_minutes(p, attempts) <= budget_minutes]
    if not fitting:
        return None
    return _pick_lints(fitting, state, now, budget_minutes, seed, samples)


_PICKERS = {
    "lints": _pick_lints,
    "shortest_duration": _pick_shortest_duration,
    "weakest_topic": _pick_weakest_topic,
    "due_first": _pick_due_first,
    "cost_aware_lints": _pick_cost_aware_lints,
    "constrained_lints": _pick_constrained_lints,
}


def pick(policy_name: str, candidates: list[dict], state: dict,
         now: float, budget_minutes: float, seed: int,
         samples: int = bandit.PROBABILITY_SAMPLES) -> dict | None:
    """One pick under a named policy, from the given candidates.

    All policies draw from the same candidate list, so comparisons
    are eligibility-matched by construction. ``budget_minutes`` is
    the remaining budget at this step: the constrained policy
    filters against it, the cost-aware policy prices duration
    against it only through expected minutes, and the other
    policies ignore it. Returns None when there is nothing to pick
    (empty candidates, or nothing fits the remaining budget).
    """
    if policy_name not in _PICKERS:
        raise ValueError(f"unknown policy: {policy_name!r}")
    if not candidates:
        return None
    return _PICKERS[policy_name](candidates, state, now,
                                 budget_minutes, seed, samples)
