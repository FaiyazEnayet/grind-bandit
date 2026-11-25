"""Deterministic, no-leak train/test splits for transfer measurement.

A delayed quiz is only evidence of transfer when the quiz item is
both unseen and built on a different solution template from the
source problem. This module classifies that relation, splits the
bank deterministically, and audits family leakage across the split:
a family-held-out split keeps whole families out of training, while
a problem-level split shares families by design and the audit says
so. Blocked cases (a one-problem family, a topic whose problems all
share one family) are reported, never silently skipped.
"""

from __future__ import annotations

import random

from grindbandit import problems

REPEAT = "repeat"
SAME_FAMILY = "same-family"
NEAR_TRANSFER = "near-transfer"
UNKNOWN = "unknown"


def classify_transfer(source_problem_id: int, quiz_problem_id: int,
                      families: dict) -> str:
    """The family relation between a source problem and a quiz item.

    ``repeat`` when the quiz item is the source problem itself,
    ``same-family`` when the two share a solution-template family,
    ``near-transfer`` when they share none, and ``unknown`` when
    either side carries no family tags.
    """
    if source_problem_id == quiz_problem_id:
        return REPEAT
    source = set(problems.families_for(source_problem_id, families))
    quiz = set(problems.families_for(quiz_problem_id, families))
    if not source or not quiz:
        return UNKNOWN
    if source & quiz:
        return SAME_FAMILY
    return NEAR_TRANSFER


def family_novelty_satisfied(source_problem_id: int,
                             quiz_problem_id: int,
                             families: dict) -> bool:
    """True only when the quiz item provably uses a different
    solution template from the source problem."""
    return classify_transfer(source_problem_id, quiz_problem_id,
                             families) == NEAR_TRANSFER


def is_novel_transfer(quiz: dict) -> bool:
    """True only for a genuinely novel transfer item.

    Both halves must hold: the quiz record must say the item was
    unseen (never attempted, never assigned) and the transfer tag
    must be near-transfer. A reused item, or a same-family item, is
    never counted as novel transfer.
    """
    assignment = quiz.get("assignment", {})
    transfer = quiz.get("transfer", {})
    return bool(assignment.get("novel")) and transfer.get("tag") == NEAR_TRANSFER


def split_by_problem(bank: list[dict], seed: int,
                     test_fraction: float = 0.25) -> dict:
    """Deterministic problem-level split: same seed, same split.

    Every problem lands on exactly one side. Families leak across
    the split by design; run audit_family_leakage to see it.
    """
    ids = sorted(p["id"] for p in bank)
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_fraction))
    test = sorted(shuffled[:n_test])
    train = sorted(shuffled[n_test:])
    return {"train": train, "test": test}


def split_by_family(bank: list[dict], families: dict, seed: int,
                    test_fraction: float = 0.25) -> dict:
    """Deterministic family-held-out split: same seed, same split.

    Whole primary families go to the test side, so no test family
    appears in training as a primary template. The primary family
    is each problem's first tag, and primary families partition the
    bank. Families are taken in seeded order until the test side
    holds at least ``test_fraction`` of the bank. Secondary tags can
    still co-occur across the split (one problem may carry two
    templates); that residual leakage is real and the audit reports
    it instead of pretending it away.
    """
    members: dict[str, list[int]] = {}
    for p in bank:
        primary = problems.primary_family(p["id"], families)
        if primary is not None:
            members.setdefault(primary, []).append(p["id"])
    family_ids = sorted(members)
    random.Random(seed).shuffle(family_ids)
    target = max(1, round(len(bank) * test_fraction))
    test_families: list[str] = []
    test_count = 0
    for fid in family_ids:
        if test_count >= target and test_families:
            break
        test_families.append(fid)
        test_count += len(members[fid])
    test_set = {pid for fid in test_families for pid in members[fid]}
    train = sorted(p["id"] for p in bank if p["id"] not in test_set)
    test = sorted(test_set)
    train_families = sorted(
        {problems.primary_family(pid, families) for pid in train}
        - {None})
    return {"train": train, "test": test,
            "train_families": train_families,
            "test_families": sorted(test_families)}


def audit_family_leakage(bank: list[dict], families: dict,
                         train_ids: list[int],
                         test_ids: list[int]) -> dict:
    """Report family leakage across a split and the blocked cases.

    ``shared_families`` names every family present on both sides.
    ``blocked`` names what cannot be split: a family with a single
    problem, and a topic whose problems all share one family, so no
    within-topic cross-family transfer exists there.
    """
    train_set = set(train_ids)
    test_set = set(test_ids)
    train_families = sorted(
        {f for pid in train_set for f in problems.families_for(pid, families)})
    test_families = sorted(
        {f for pid in test_set for f in problems.families_for(pid, families)})
    blocked: list[dict] = []
    members: dict[str, list[int]] = {}
    for row in families.get("assignments", []):
        for fam in row.get("families", []):
            members.setdefault(fam, []).append(row["problem_id"])
    for fid in sorted(members):
        if len(members[fid]) == 1:
            blocked.append({"kind": "single_problem_family",
                            "family": fid,
                            "problem_id": members[fid][0]})
    by_topic: dict[str, list[dict]] = {}
    for p in bank:
        for topic in p["topics"]:
            by_topic.setdefault(topic, []).append(p)
    for topic in sorted(by_topic):
        union = {f for p in by_topic[topic]
                 for f in problems.families_for(p["id"], families)}
        if len(union) == 1:
            blocked.append({"kind": "single_family_topic",
                            "topic": topic,
                            "family": sorted(union)[0]})
    return {"train_families": train_families,
            "test_families": test_families,
            "shared_families": sorted(set(train_families) & set(test_families)),
            "train_only_families": sorted(
                set(train_families) - set(test_families)),
            "test_only_families": sorted(
                set(test_families) - set(train_families)),
            "blocked": blocked}
