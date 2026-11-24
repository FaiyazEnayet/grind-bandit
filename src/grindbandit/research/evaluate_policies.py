"""Evaluate the shadow policy branch on a real state file.

Reads only: this counts linked delayed-transfer observations in the
attempt log and reports the real-data gate verdict from
``grindbandit.research.policy_comparison``. It never writes state,
never touches mastery, the bandit, or rewards, and its output is a
report, not a policy promotion.

With ``--simulate`` it also runs the fixed 15, 30, and 60 minute
simulation comparison. Simulation validates the machinery only; a
simulation winner is never evidence about real users and never
promotes a policy.

Run as a script: ``python -m grindbandit.research.evaluate_policies
--state PATH`` prints the report as JSON. With ``--demo`` it evaluates
the scripted demo history instead of a file; that history is scripted,
not real practice, so its verdict is labeled as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grindbandit.research import policy_comparison as pc


def evaluate(state: dict, data_source: str = "state file") -> dict:
    """The real-data gate verdict for a state dict, labeled with
    where the state came from."""
    report = pc.evaluate(state)
    report["data_source"] = data_source
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the shadow policy branch on a state file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--state", type=Path, help="path to a state JSON file")
    group.add_argument("--demo", action="store_true",
                       help="evaluate the scripted demo history")
    parser.add_argument("--simulate", action="store_true",
                        help="also run the simulation comparison")
    parser.add_argument("--replications", type=int, default=30,
                        help="simulation replications per budget")
    args = parser.parse_args(argv)
    if args.demo:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
        from common import build_demo_state

        report = evaluate(build_demo_state(42),
                          data_source="scripted demo history")
    else:
        with open(args.state, encoding="utf-8") as fh:
            report = evaluate(json.load(fh),
                              data_source=f"state file {args.state}")
    if args.simulate:
        report["simulation"] = pc.compare(replications=args.replications)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
