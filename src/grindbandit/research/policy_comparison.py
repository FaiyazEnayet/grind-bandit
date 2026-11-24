"""Budget-aware policy comparison under the GB5 transfer proxy.

The production selector is current LinTS. This module asks, offline,
whether any budget-aware challenger would serve a learner better,
and it answers in two parts that must never be confused.

Real data first. A challenger is adopted only after real measurement
shows improvement on the delayed-transfer proxy: attempts joined to
their delayed transfer quizzes by immutable attempt id, the GB5
laboratory. The gate needs at least 30 linked observations. The
scripted demo history holds 30 attempts and no delayed quizzes at
all, so the verdict on the data available here is NOT IDENTIFIABLE
and the simple model stays. ``evaluate`` reports exactly that.

Simulation second. ``compare`` runs every policy through fixed 15,
30, and 60 minute sessions in a simulated world with a known ground
truth, and reports transfer, overrun, Hard share, and topic coverage
per policy per budget. Simulation validates the machinery only: it
can surface a challenger that wins inside the simulated world, and
it can show the gate blocking a winner that starves Hard topics or
overruns its budget, but a simulation win is never evidence about
real users and never promotes a policy. Simulated rows are labeled
as such by construction: they never touch a real state file.

The simulated world, predeclared before any result was looked at:
- The learner has a true mastery per topic, strong on array and
  string (0.8), weak on dp, graph, and backtracking (0.3), 0.6
  elsewhere, mirroring the scripted demo history's topic strengths.
- A problem takes a true duration drawn around its difficulty mean
  (Easy 12, Medium 25, Hard 45 minutes, with noise, clipped to
  3-90). Every policy in one replication faces the same true
  durations, so duration differences cannot favor one policy.
- The chance of solving is sigmoid(true mastery minus a difficulty
  offset: Easy -1, Medium 0, Hard +1).
- Learning is match-gated: a solve adds 0.30 times the difficulty
  match times the headroom (1 minus mastery) to each topic on the
  problem, a failure adds 0.06 times the same. Match is 1 minus the
  gap between the problem's difficulty and the difficulty the
  learner's mastery implies, using the same target rule live
  selection uses, so a Hard problem on a weak topic teaches nothing
  and an Easy problem on a weak topic teaches the most per attempt.
- Transfer is measured on novel quiz problems, up to two per topic,
  fixed before any session runs so every policy in one replication
  is scored on the same set: the transfer proxy is the mean solve
  probability over that set after the session. A quiz problem a
  policy practiced during its session stops being a transfer test
  for that policy and leaves its quiz set.
- Sessions start from a scripted 30-attempt history over 13 days,
  logged through the real engine, so modeled mastery, the bandit
  posterior, and historically observed minutes all exist before the
  first pick, and every policy draws from the same eligible set
  under the live eligibility rule.

The gate, also predeclared: a challenger is adoptable only when its
transfer improvement over LinTS has a paired bootstrap 95 percent
interval entirely above zero, its mean overrun is at most 5 percent
of the budget, and it does not starve Hard topics (Hard share below
a quarter of LinTS) or narrow topic coverage (below half of LinTS).
A simulation winner needs that gate in a majority of budgets, and
even then it stays shadow until real measurement agrees.
"""

from __future__ import annotations

import copy
import math
import random
import statistics
from collections import Counter

from grindbandit import bandit, mastery, problems
from grindbandit import state as st
from grindbandit.cli import select_candidates
from grindbandit.policies import registry as reg
from grindbandit.policies import time_aware as ta

# --- Predeclared gate constants (set before any data is looked at) --------

# Minimum linked delayed-transfer observations for the real-data
# gate to say anything at all. The same threshold the GB5 laboratory
# uses: a rank comparison on fewer points is too noisy to promote a
# policy on, so below it the honest verdict is NOT IDENTIFIABLE.
MIN_LINKED_OBSERVATIONS = 30

# Fixed session budgets under test, in minutes.
BUDGETS = (15, 30, 60)

# A policy starves Hard topics when its Hard pick share falls below
# this fraction of the LinTS Hard share in the same budget, and it
# narrows coverage when its distinct-topic count falls below this
# fraction of the LinTS count.
STARVATION_FRACTION = 0.25
COVERAGE_FRACTION = 0.5

# A policy overruns when its mean overrun exceeds this fraction of
# the budget. The constrained policy is built never to pick over
# budget, but true durations are noisy, so the gate tolerates a
# small mean overrun rather than demanding an impossible zero.
OVERRUN_TOLERANCE_FRACTION = 0.05

# Paired bootstrap resamples for the transfer difference interval.
BOOTSTRAP_SAMPLES = 2000

# --- Simulated world truth (predeclared, see the module docstring) --------

STRONG_TOPICS = {"array", "string"}
WEAK_TOPICS = {"dp", "graph", "backtracking"}
STRONG_MASTERY = 0.8
WEAK_MASTERY = 0.3
DEFAULT_MASTERY = 0.6

DIFFICULTY_OFFSET = {"Easy": -1.0, "Medium": 0.0, "Hard": 1.0}
DURATION_MEAN_SD = {"Easy": (12.0, 4.0), "Medium": (25.0, 7.0),
                    "Hard": (45.0, 10.0)}
DURATION_CLIP = (3.0, 90.0)
SOLVE_GAIN = 0.30
FAIL_GAIN = 0.06

_SIMULATION_NOW = 1_750_000_000.0
_HISTORY_DAYS = 13.0
_HISTORY_ATTEMPTS = 30
_QUIZ_PER_TOPIC = 2
_MAX_PICKS_PER_SESSION = 20
_SECONDS_PER_DAY = 86400.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def solve_probability(mean_mastery: float, difficulty: str) -> float:
    """True chance of solving: sigmoid of mastery minus the
    difficulty offset."""
    return sigmoid(mean_mastery - DIFFICULTY_OFFSET[difficulty])


def true_duration_minutes(rng: random.Random, difficulty: str) -> float:
    """One true duration draw for a problem of this difficulty."""
    mean, sd = DURATION_MEAN_SD[difficulty]
    value = rng.gauss(mean, sd)
    return min(max(value, DURATION_CLIP[0]), DURATION_CLIP[1])


def new_learner(topics) -> dict:
    """A fresh simulated learner: true mastery per topic, strong
    where the demo history is strong, weak where it is weak."""
    mastery_map = {}
    for t in topics:
        if t in STRONG_TOPICS:
            mastery_map[t] = STRONG_MASTERY
        elif t in WEAK_TOPICS:
            mastery_map[t] = WEAK_MASTERY
        else:
            mastery_map[t] = DEFAULT_MASTERY
    return {"mastery": mastery_map, "attempted": set()}


def apply_learning(learner: dict, problem: dict, solved: bool) -> None:
    """Move the learner's true mastery after one attempt.

    The gain is match-gated: 1 minus the gap between the problem's
    difficulty and the difficulty the learner's mastery implies,
    times the headroom. A solve gains SOLVE_GAIN times the match,
    a failure FAIL_GAIN times the match.
    """
    topics = problem["topics"]
    mean = sum(learner["mastery"][t] for t in topics) / len(topics)
    target = bandit.target_difficulty(mean)
    match = 1.0 - abs(bandit.DIFFICULTY_NUM[problem["difficulty"]] - target)
    rate = SOLVE_GAIN if solved else FAIL_GAIN
    for t in topics:
        learner["mastery"][t] += rate * match * (1.0 - learner["mastery"][t])
    learner["attempted"].add(problem["id"])


def _log_attempt(state: dict, problem: dict, solved: bool, minutes: int,
                 now: float, attempt_id: str) -> None:
    """One attempt through the real engine, as the CLI logs it:
    context, mastery update, model-derived reward, posterior update,
    and the append-only log entry."""
    x = bandit.context(problem, state, now)
    tm = {t: mastery.mean(state, t, now) for t in problem["topics"]}
    weak = bandit.weakest_topic(problem, tm)
    before = tm[weak]
    mastery.update(state, problem["topics"], solved, now)
    after = mastery.mean(state, weak, now)
    reward = max(-1.0, min(1.0, after - before))
    bandit.observe(state, x, reward)
    st.record_attempt(state, problem, solved, minutes, reward, now,
                      attempt_id=attempt_id)


def build_world(seed: int):
    """One simulated world: the real bank, a scripted 30-attempt
    history over 13 days logged through the real engine, and the
    true learner whose mastery that history moved.

    Returns (bank, state, learner, now). The history's minutes
    follow the world's duration model, so expected minutes from
    history are informative about difficulty instead of uniform.
    """
    rng = random.Random(seed)
    dur_rng = random.Random(seed + 10_000)
    bank = problems.load()
    topics = problems.all_topics(bank)
    learner = new_learner(topics)
    state: dict = {"topics": {}, "problems": {}, "log": [], "bandit": {}}
    now = _SIMULATION_NOW
    start = now - _HISTORY_DAYS * _SECONDS_PER_DAY
    order = bank[:]
    rng.shuffle(order)
    for n, p in enumerate(order[:_HISTORY_ATTEMPTS]):
        ts = start + n * (_HISTORY_DAYS * _SECONDS_PER_DAY
                          / _HISTORY_ATTEMPTS)
        mean = sum(learner["mastery"][t] for t in p["topics"]) / len(p["topics"])
        solved = rng.random() < solve_probability(mean, p["difficulty"])
        minutes = int(round(true_duration_minutes(dur_rng, p["difficulty"])))
        _log_attempt(state, p, solved, minutes, ts,
                     attempt_id=f"att-sim-{seed}-{n + 1:03d}")
        apply_learning(learner, p, solved)
    return bank, state, learner, now


def quiz_set(bank: list[dict], state: dict,
             per_topic: int = _QUIZ_PER_TOPIC) -> list[dict]:
    """Novel quiz problems, up to ``per_topic`` per topic, lowest id
    first. Novel means never attempted in the state log. Fix the set
    before sessions run so every policy is scored on the same set."""
    attempted = {e.get("problem_id") for e in state.get("log") or []}
    topics = sorted({t for p in bank for t in p["topics"]})
    chosen: dict[int, dict] = {}
    for topic in topics:
        fresh = [p for p in bank
                 if topic in p["topics"] and p["id"] not in attempted]
        for p in fresh[:per_topic]:
            chosen[p["id"]] = p
    return [chosen[pid] for pid in sorted(chosen)]


def transfer_proxy(learner: dict, quiz: list[dict]) -> float:
    """Mean solve probability over the quiz set: the transfer proxy
    for this learner after its session. An expectation over the
    known truth, so the comparison is paired and low-variance."""
    if not quiz:
        return 0.0
    total = 0.0
    for p in quiz:
        mean = sum(learner["mastery"][t] for t in p["topics"]) / len(p["topics"])
        total += solve_probability(mean, p["difficulty"])
    return total / len(quiz)


def run_session(policy_name: str, bank: list[dict], state: dict,
                learner: dict, now: float, budget_minutes: float,
                seed: int, samples: int = 1000,
                quiz: list[dict] | None = None) -> dict:
    """One simulated session under one policy.

    Picks from the live eligible set until the budget is spent or
    nothing is pickable. The attempt costs its true duration, which
    is fixed per replication across policies, so an overrun is a
    real overspend, not a different draw. The state and learner are
    copied first, so one policy's session never moves another's
    starting point. Returns the session report.
    """
    state = copy.deepcopy(state)
    learner = copy.deepcopy(learner)
    rng = random.Random(seed)
    dur_rng = random.Random(seed + 999)
    durations = {p["id"]: true_duration_minutes(dur_rng, p["difficulty"])
                 for p in bank}
    if quiz is None:
        quiz = quiz_set(bank, state)
    spent = 0.0
    picks: list[int] = []
    step = 0
    while spent < budget_minutes and step < _MAX_PICKS_PER_SESSION:
        cands, _ = select_candidates(bank, state, now)
        if not cands:
            break
        picked = ta.pick(policy_name, cands, state, now,
                         budget_minutes - spent, seed=seed + step,
                         samples=samples)
        if picked is None:
            break
        true_min = durations[picked["id"]]
        spent += true_min
        mean = sum(learner["mastery"][t] for t in picked["topics"]) \
            / len(picked["topics"])
        solved = rng.random() < solve_probability(mean, picked["difficulty"])
        _log_attempt(state, picked, solved, int(round(true_min)), now,
                     attempt_id=f"att-sim-s{seed}-{step}")
        apply_learning(learner, picked, solved)
        picks.append(picked["id"])
        step += 1
    picked_set = set(picks)
    scored_quiz = [q for q in quiz if q["id"] not in picked_set]
    by_id = {p["id"]: p for p in bank}
    hard = sum(1 for pid in picks if by_id[pid]["difficulty"] == "Hard")
    covered = {t for pid in picks for t in by_id[pid]["topics"]}
    return {
        "policy": policy_name,
        "budget_minutes": float(budget_minutes),
        "spent_minutes": spent,
        "overrun_minutes": max(0.0, spent - budget_minutes),
        "n_picks": len(picks),
        "picks": picks,
        "hard_share": (hard / len(picks)) if picks else 0.0,
        "topics_covered": len(covered),
        "transfer_proxy": transfer_proxy(learner, scored_quiz),
        "n_quiz": len(scored_quiz),
    }


def _paired_bootstrap_ci(diffs: list[float], seed: int,
                         samples: int = BOOTSTRAP_SAMPLES):
    """Paired bootstrap 95 percent interval for the mean of
    per-replication differences. Deterministic for a fixed seed."""
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choices(diffs, k=n)) / n)
    means.sort()
    low = means[int(0.025 * samples)]
    high = means[min(int(0.975 * samples), samples - 1)]
    return low, high


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def compare(budgets=BUDGETS, replications: int = 30, base_seed: int = 7,
            samples: int = 1000) -> dict:
    """Compare every policy at every budget over replications of the
    simulated world. Each replication builds one world and runs all
    six policies from identical starting conditions, so differences
    are paired by replication.

    The report carries one row per policy per budget, the per-budget
    winner under the predeclared gate, and the simulation winner.
    The real-data verdict is NOT IDENTIFIABLE throughout: this
    comparison is machinery validation, and nothing in it changes
    what real measurement has or has not shown.
    """
    rows: list[dict] = []
    per_budget_winner: dict[int, str | None] = {}
    wins: Counter = Counter()
    for budget in budgets:
        per_policy: dict[str, list[dict]] = {name: [] for name in reg.POLICY_NAMES}
        for rep in range(replications):
            seed = base_seed + rep
            bank, state, learner, now = build_world(seed)
            quiz = quiz_set(bank, state)
            for name in reg.POLICY_NAMES:
                per_policy[name].append(
                    run_session(name, bank, state, learner, now, budget,
                                seed=seed, samples=samples, quiz=quiz))
        base = per_policy["lints"]
        base_transfer = [s["transfer_proxy"] for s in base]
        base_hard = _mean([s["hard_share"] for s in base])
        base_cov = _mean([s["topics_covered"] for s in base])
        best: str | None = None
        best_diff = 0.0
        for name in reg.POLICY_NAMES:
            sessions = per_policy[name]
            transfers = [s["transfer_proxy"] for s in sessions]
            diffs = [a - b for a, b in zip(transfers, base_transfer)]
            ci_low, ci_high = _paired_bootstrap_ci(diffs, seed=base_seed)
            overrun_mean = _mean([s["overrun_minutes"] for s in sessions])
            overrun_max = max(s["overrun_minutes"] for s in sessions)
            overrun_flag = overrun_mean > OVERRUN_TOLERANCE_FRACTION * budget
            hard = _mean([s["hard_share"] for s in sessions])
            cov = _mean([s["topics_covered"] for s in sessions])
            starvation_flag = (
                (base_hard > 0.01 and hard < STARVATION_FRACTION * base_hard)
                or (base_cov > 0 and cov < COVERAGE_FRACTION * base_cov)
            )
            diff_mean = _mean(diffs)
            rows.append({
                "policy": name,
                "budget": budget,
                "transfer_proxy_mean": _mean(transfers),
                "transfer_proxy_sd": _stdev(transfers),
                "transfer_diff_vs_lints_mean": diff_mean,
                "transfer_diff_ci_low": ci_low,
                "transfer_diff_ci_high": ci_high,
                "spent_mean": _mean([s["spent_minutes"] for s in sessions]),
                "n_picks_mean": _mean([s["n_picks"] for s in sessions]),
                "overrun_mean": overrun_mean,
                "overrun_max": overrun_max,
                "overrun_flag": bool(overrun_flag),
                "hard_share": hard,
                "topics_covered_mean": cov,
                "starvation_flag": bool(starvation_flag),
                "n_replications": replications,
            })
            if (name != "lints" and ci_low > 0.0
                    and not overrun_flag and not starvation_flag
                    and diff_mean > best_diff):
                best = name
                best_diff = diff_mean
        per_budget_winner[budget] = best
        if best is not None:
            wins[best] += 1
    simulation_winner = None
    for name, count in wins.items():
        if count > len(list(budgets)) // 2:
            simulation_winner = name
    return {
        "policy_version": reg.POLICY_VERSION,
        "budgets": list(budgets),
        "replications": replications,
        "base_seed": base_seed,
        "rows": rows,
        "per_budget_winner": per_budget_winner,
        "simulation_winner": simulation_winner,
        "simulation_note": (
            "Simulation validates the comparison machinery only. A "
            "simulation winner is a win inside the simulated world, "
            "never evidence about real users, and never promotes a "
            "policy: adoption needs real measurement."
        ),
        "real_data_verdict": "NOT_IDENTIFIABLE",
        "real_data_note": (
            "No real-data evaluation is part of this comparison. The "
            "standing real-data verdict is NOT IDENTIFIABLE until "
            "30 linked delayed-transfer observations exist."
        ),
    }


def gate_check(challenger_row: dict, lints_row: dict) -> dict:
    """The predeclared adoption gate for one challenger row against
    the LinTS row in the same budget: transfer improvement whose
    paired interval excludes zero, no overrun, no starvation. Each
    failure alone blocks adoption."""
    failures: list[str] = []
    if not challenger_row["transfer_diff_ci_low"] > 0.0:
        failures.append(
            "transfer: improvement over LinTS is not robust, the "
            "paired interval does not exclude zero")
    if challenger_row["overrun_flag"]:
        failures.append(
            "overrun: mean overrun exceeds the tolerated share of "
            "the stated budget")
    if challenger_row["starvation_flag"]:
        failures.append(
            "starvation: Hard share or topic coverage falls below "
            "the tolerated share of LinTS")
    return {"adoptable": not failures, "failures": failures}


def extract_linked(state: dict) -> int:
    """Count completed linked observations in a real state: delayed
    quizzes with a recorded outcome whose source attempt is in the
    log. Missing or unrecorded outcomes are not counted as anything.
    """
    log_by_attempt = {e.get("attempt_id"): e
                      for e in state.get("log") or []
                      if e.get("attempt_id") is not None}
    n = 0
    for quiz in state.get("delayed_quizzes") or []:
        outcome = quiz.get("outcome") or {}
        if "solved" not in outcome:
            continue
        if quiz.get("source_attempt_id") in log_by_attempt:
            n += 1
    return n


def evaluate(state: dict) -> dict:
    """The real-data gate on a state dict. Read-only: the state is
    untouched. Below 30 linked delayed-transfer observations the
    verdict is NOT IDENTIFIABLE, which is a finding about the data,
    not a failure of the comparison."""
    n_linked = extract_linked(state)
    verdict = ("NOT_IDENTIFIABLE"
               if n_linked < MIN_LINKED_OBSERVATIONS else "IDENTIFIABLE")
    return {
        "verdict": verdict,
        "n_linked_observations": n_linked,
        "min_linked_observations": MIN_LINKED_OBSERVATIONS,
        "n_attempts": len(state.get("log") or []),
        "production_policy": reg.active_production_policy(),
        "policy_version": reg.POLICY_VERSION,
        "reason": (
            "Fewer than 30 linked delayed-transfer observations, so "
            "no policy comparison can be measured on real data yet."
            if verdict == "NOT_IDENTIFIABLE" else
            "Enough linked delayed-transfer observations to run the "
            "real-data comparison."
        ),
    }
