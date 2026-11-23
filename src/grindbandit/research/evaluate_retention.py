"""Evaluate the shadow retention model on a real state file.

Reads only: this builds retention observations from the attempt log,
runs the predeclared temporal holdout comparison from
``grindbandit.research.retention``, and reports the gate verdict. It
never writes state, never touches mastery, the bandit, or rewards,
and its output is a report, not a model promotion.

An observation is a later attempt on a topic after an earlier solved
attempt on the same topic: the elapsed days between the two attempts
and whether the later attempt was solved. Attempts with no earlier
attempt on that topic, and attempts whose earlier attempt on the topic
was a failure, yield no observation, because the model predicts recall
of learned material. One attempt on a multi-topic problem yields one
observation per topic.

Run as a script: ``python -m grindbandit.research.evaluate_retention
--state PATH`` prints the report as JSON. With ``--demo`` it evaluates
the scripted demo history instead of a file; that history is scripted,
not real practice, so its verdict is labeled as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grindbandit.research import retention as rt

_SECONDS_PER_DAY = 86400.0


def extract_observations(state: dict) -> dict:
    """Build retention observations from the attempt log of a state dict.

    Returns the observations plus counts of what was skipped and why,
    so a small or empty result can be told apart from a broken log.
    """
    log = sorted(state.get("log") or [], key=lambda e: e.get("ts") or 0.0)
    last: dict[str, dict] = {}
    solved_counts: dict[str, int] = {}
    observations: list[dict] = []
    n_no_prior = 0
    n_prior_failed = 0
    for entry in log:
        ts = entry.get("ts")
        if ts is None:
            continue
        solved = bool(entry.get("solved"))
        for topic in entry.get("topics") or []:
            prior = last.get(topic)
            if prior is None:
                n_no_prior += 1
            elif not prior["solved"]:
                n_prior_failed += 1
            else:
                observations.append({
                    "topic": topic,
                    "delta_days": (ts - prior["ts"]) / _SECONDS_PER_DAY,
                    "solved": solved,
                    "prior_successes": solved_counts.get(topic, 0),
                    "ts": ts,
                    "last_ts": prior["ts"],
                })
            if solved:
                solved_counts[topic] = solved_counts.get(topic, 0) + 1
            last[topic] = {"ts": ts, "solved": solved}
    return {
        "observations": observations,
        "n_observations": len(observations),
        "n_attempts": len(log),
        "n_skipped_no_prior_attempt": n_no_prior,
        "n_skipped_prior_attempt_failed": n_prior_failed,
    }


def evaluate(state: dict, data_source: str = "state file") -> dict:
    """Run the retention gate on one state dict and return the report."""
    extracted = extract_observations(state)
    observations = extracted["observations"]
    train, holdout = rt.split_holdout(observations)
    comparison = rt.compare_models(train, holdout)
    return {
        "data_source": data_source,
        "production_model": rt.active_production_model(),
        "registry": rt.registry_summary(),
        "n_attempts": extracted["n_attempts"],
        "n_observations": extracted["n_observations"],
        "n_skipped_no_prior_attempt": extracted["n_skipped_no_prior_attempt"],
        "n_skipped_prior_attempt_failed": extracted["n_skipped_prior_attempt_failed"],
        "n_train": comparison["n_train"],
        "n_holdout": comparison["n_holdout"],
        "models": comparison["models"],
        "gate_failures": comparison.get("gate_failures", {}),
        "verdict": comparison["verdict"],
        "reason": comparison["reason"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the shadow retention model on a state file.")
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
