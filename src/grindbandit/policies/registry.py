"""Versioned policy branch registry.

The live selector is still current LinTS and nothing here changes
it. Every challenger policy lives under ``POLICY_VERSION`` as a
shadow candidate: it can be compared offline against the production
policy, but it is never wired into live selection until real
measurement passes the gate in ``grindbandit.research.policy_comparison``
(improvement on the delayed-transfer proxy, no unacceptable
starvation of Hard and rare topics, no overrun relative to the stated
budget). Until then the registry is the record that these policies
are proposals, not behavior.
"""

from __future__ import annotations

# Version of the policy branch. Bump this whenever a policy's
# definition changes, so a comparison report stays interpretable;
# a report names the version it ran under.
POLICY_VERSION = "policy-v1"

POLICY_NAMES = ["lints", "shortest_duration", "weakest_topic",
                "due_first", "cost_aware_lints", "constrained_lints"]

REGISTRY = {
    "lints": {
        "status": "production",
        "description": "Current Thompson sampling over the linear "
                       "context model (bandit.build_recommendation).",
    },
    "shortest_duration": {
        "status": "shadow",
        "description": "Shortest expected duration first, from "
                       "leave-one-problem-out difficulty means.",
    },
    "weakest_topic": {
        "status": "shadow",
        "description": "Largest gap on the problem's weakest topic "
                       "first, ignoring duration.",
    },
    "due_first": {
        "status": "shadow",
        "description": "Due spaced reviews first, ordered by interval "
                       "since the target skill was practiced, then new "
                       "practice by the same interval.",
    },
    "cost_aware_lints": {
        "status": "shadow",
        "description": "Contextual cost-aware LinTS: the sampled LinTS "
                       "score minus lambda times expected minutes.",
    },
    "constrained_lints": {
        "status": "shadow",
        "description": "LinTS among problems whose expected minutes "
                       "fit the remaining budget; no pick at all when "
                       "nothing fits, instead of overrunning.",
    },
}


def active_production_policy() -> str:
    """The policy live selection actually uses. Only one policy is
    ever production, and it is the current LinTS selector."""
    for name, entry in REGISTRY.items():
        if entry["status"] == "production":
            return name
    raise ValueError("no production policy in the registry")
