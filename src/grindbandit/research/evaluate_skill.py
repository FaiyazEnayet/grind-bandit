"""Evaluate the shadow skill model on a real state file.

Reads only: this builds next-attempt observations from the attempt
log, runs the predeclared temporal holdout comparison from
``grindbandit.research.skill_model``, and reports the gate verdict.
It never writes state, never touches mastery, the bandit, or rewards,
and its output is a report, not a model promotion.

An observation is one attempt: problem id, difficulty, topics,
whether it was solved, and its timestamp. Attempts on a problem
already attempted before are marked as repeats. Attempts are
user-selected (bandit picks or manual choice), not randomly assigned
problems, so the comparison is a predictive check, not an estimate
of true ability: a single student and a sparse bank cannot identify
ability, and the report says so.

Run as a script: ``python -m grindbandit.research.evaluate_skill
--state PATH`` prints the report as JSON. With ``--demo`` it evaluates
the scripted demo history instead of a file; that history is scripted,
not real practice, so its verdict is labeled as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grindbandit.research import skill_model as sm

_SECONDS_PER_DAY = 86400.0

SELECTIVITY_NOTE = (
    "Attempts are user-selected (bandit picks or manual choice), not "
    "randomly assigned problems, so this comparison is a predictive "
    "check: it cannot identify true ability from one student and a "
    "sparse bank, and a learned head that wins here is still a shadow "
    "model until a human decides otherwise on real-data evidence."
)


def extract_observations(state: dict) -> dict:
    """Build next-attempt observations from the attempt log of a state dict.

    Returns the observations plus counts of what was skipped and why,
    so a small or empty result can be told apart from a broken log.
    Repeated attempts on the same problem are counted, not dropped:
    predicting the next attempt on a familiar problem is part of the task.
    """
    log = sorted(state.get("log") or [], key=lambda e: e.get("ts") or 0.0)
    seen: dict[int, int] = {}
    observations: list[dict] = []
    n_no_timestamp = 0
    n_repeats = 0
    for index, entry in enumerate(log):
        ts = entry.get("ts")
        if ts is None:
            n_no_timestamp += 1
            continue
        problem_id = entry.get("problem_id")
        prior = seen.get(problem_id, 0)
        seen[problem_id] = prior + 1
        if prior > 0:
            n_repeats += 1
        observations.append({
            "problem_id": problem_id,
            "difficulty": entry.get("difficulty"),
            "topics": list(entry.get("topics") or []),
            "solved": bool(entry.get("solved")),
            "ts": ts,
            "attempt_index": index,
            "prior_attempts_on_problem": prior,
            "repeat": prior > 0,
        })
    return {
        "observations": observations,
        "n_observations": len(observations),
        "n_attempts": len(log),
        "n_repeats": n_repeats,
        "n_skipped_no_timestamp": n_no_timestamp,
    }


def evaluate(state: dict, data_source: str = "state file") -> dict:
    """Run the skill-model gate on one state dict and return the report."""
    extracted = extract_observations(state)
    observations = extracted["observations"]
    train, holdout = sm.split_holdout(observations)
    comparison = sm.compare_models(train, holdout)
    return {
        "data_source": data_source,
        "production_model": sm.active_production_model(),
        "registry": sm.registry_summary(),
        "selectivity_note": SELECTIVITY_NOTE,
        "n_attempts": extracted["n_attempts"],
        "n_observations": extracted["n_observations"],
        "n_repeats": extracted["n_repeats"],
        "n_skipped_no_timestamp": extracted["n_skipped_no_timestamp"],
        "n_train": comparison["n_train"],
        "n_holdout": comparison["n_holdout"],
        "best_baseline": comparison.get("best_baseline"),
        "models": comparison["models"],
        "gate_failures": comparison.get("gate_failures", {}),
        "verdict": comparison["verdict"],
        "reason": comparison["reason"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the shadow skill model on a state file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--state", type=Path, help="path to a state JSON file")
    group.add_argument("--demo", action="store_true",
                       help="evaluate the scripted demo history")
    args = parser.parse_args(argv)
    if args.demo:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
        from common import build_demo_state

        report = evaluate(build_demo_state(42), data_source="scripted demo history")
    else:
        with open(args.state, encoding="utf-8") as fh:
            report = evaluate(json.load(fh), data_source=f"state file {args.state}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
