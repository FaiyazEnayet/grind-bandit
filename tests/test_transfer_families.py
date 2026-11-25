"""Stage 3 GB16: transfer-aware problem families and novelty controls.

A different problem ID does not prove novel transfer: interview
questions often share a solution template, so a delayed quiz can hand
out a problem whose template matches the source problem exactly and a
solved quiz would then reflect template recall rather than transfer.

This gates the curated, versioned family layer: at least 10 solution
template families with stable identifiers, multi-label tags per
problem, a quiz chooser that prefers the same target skill with a
different surface item AND a different family where one is available,
a transfer record on every quiz, and a deterministic no-leak split
with a family-leakage audit. A repeat is never counted as novel
transfer: novelty needs both an unseen item and a different family.
"""

import copy

import pytest

from grindbandit import problems, state
from grindbandit.research import transfer_splits as ts


def _problem(pid, topics=("array",)):
    return {"id": pid, "title": f"P{pid}", "difficulty": "Easy",
            "topics": list(topics)}


def _state():
    return {"topics": {}, "problems": {}, "log": [], "bandit": {},
            "delayed_quizzes": []}


def _toy_families():
    """Two families over five problems on one topic, for exact
    known-answer fixtures that do not depend on the curation."""
    return {
        "version": "families-v1",
        "families": [
            {"id": "fam-a", "label": "Family A",
             "description": "Template A.", "examples": [1]},
            {"id": "fam-b", "label": "Family B",
             "description": "Template B.", "examples": [2]},
        ],
        "assignments": [
            {"problem_id": 1, "families": ["fam-a"]},
            {"problem_id": 2, "families": ["fam-b"]},
            {"problem_id": 3, "families": ["fam-a"]},
            {"problem_id": 4, "families": ["fam-b"]},
            {"problem_id": 5, "families": ["fam-a"]},
        ],
    }


def test_stage3_grind_bandit_16_families_file_valid():
    """The shipped layer is versioned, covers the whole bank, and
    every reference it makes resolves."""
    bank = problems.load()
    fams = problems.load_families()
    assert fams["version"] == "families-v1"
    ids = [f["id"] for f in fams["families"]]
    assert len(ids) >= 10
    assert len(set(ids)) == len(ids)
    errors = problems.validate_families(bank, fams)
    assert errors == []
    assigned = {a["problem_id"] for a in fams["assignments"]}
    assert assigned == {p["id"] for p in bank}


def test_stage3_grind_bandit_16_bfs_families_distinct():
    """BFS flood fill and BFS shortest path are different families:
    Number of Islands is flood fill, Rotting Oranges and Shortest
    Path in Binary Matrix are shortest path."""
    fams = problems.load_families()
    assert "bfs-flood-fill" in problems.families_for(200, fams)
    assert "bfs-shortest-path" not in problems.families_for(200, fams)
    assert "bfs-shortest-path" in problems.families_for(994, fams)
    assert "bfs-flood-fill" not in problems.families_for(994, fams)
    assert "bfs-shortest-path" in problems.families_for(1091, fams)


def test_stage3_grind_bandit_16_binary_search_families_distinct():
    """Binary search on the answer and binary search on an index are
    different families: Koko Eating Bananas searches the answer,
    Search in Rotated Sorted Array searches an index."""
    fams = problems.load_families()
    assert "binary-search-answer" in problems.families_for(875, fams)
    assert "binary-search-index" not in problems.families_for(875, fams)
    assert "binary-search-index" in problems.families_for(33, fams)
    assert "binary-search-answer" not in problems.families_for(33, fams)


def test_stage3_grind_bandit_16_chooser_prefers_near_transfer():
    """Exact fixture: source in fam-a, candidates in both families.
    The chooser picks the lowest-id near-transfer candidate (2),
    not the lowest-id candidate overall in the same family (3)."""
    fams = _toy_families()
    bank = [_problem(i) for i in (1, 2, 3, 4, 5)]
    source = _problem(1)
    pick = state.choose_quiz_problem(bank, source, "array", _state(),
                                     families=fams)
    assert pick["id"] == 2


def test_stage3_grind_bandit_16_chooser_novelty_still_first():
    """Family preference is secondary to novelty: a novel same-family
    problem still beats an already-attempted near-transfer one."""
    fams = _toy_families()
    bank = [_problem(i) for i in (1, 2, 3, 4, 5)]
    source = _problem(1)
    st = _state()
    st["problems"] = {"2": {}, "4": {}}
    pick = state.choose_quiz_problem(bank, source, "array", st,
                                     families=fams)
    assert pick["id"] == 3


def test_stage3_grind_bandit_16_chooser_reports_missing_alternative():
    """When the pool holds only same-family candidates, the chooser
    still picks one, and the quiz record says the near-transfer
    alternative was missing instead of staying silent."""
    fams = _toy_families()
    bank = [_problem(i) for i in (1, 3, 5)]
    source = _problem(1)
    st = _state()
    quiz = state.schedule_delayed_quiz(st, bank, source, "array", 1000.0,
                                       families=fams)
    assert quiz["quiz_problem_id"] == 3
    transfer = quiz["transfer"]
    assert transfer["tag"] == "same-family"
    assert transfer["family_novelty_satisfied"] is False
    assert transfer["near_transfer_available"] is False
    assert transfer["source_families"] == ["fam-a"]
    assert transfer["quiz_families"] == ["fam-a"]


def test_stage3_grind_bandit_16_quiz_transfer_record():
    """A quiz with a near-transfer candidate available records the
    transfer tag, both family sets, and that novelty is satisfied."""
    fams = _toy_families()
    bank = [_problem(i) for i in (1, 2, 3, 4, 5)]
    source = _problem(1)
    st = _state()
    quiz = state.schedule_delayed_quiz(st, bank, source, "array", 1000.0,
                                       families=fams)
    assert quiz["quiz_problem_id"] == 2
    transfer = quiz["transfer"]
    assert transfer["tag"] == "near-transfer"
    assert transfer["family_novelty_satisfied"] is True
    assert transfer["near_transfer_available"] is True
    assert transfer["source_families"] == ["fam-a"]
    assert transfer["quiz_families"] == ["fam-b"]
    assert quiz["assignment"]["novel"] is True
    assert ts.is_novel_transfer(quiz) is True


def test_stage3_grind_bandit_16_no_silent_repeat_as_novel_transfer():
    """A repeat is never a novel transfer: the same problem as the
    source classifies as repeat, and a reused quiz item (already
    attempted or assigned) is not a novel transfer even when its
    family differs from the source."""
    fams = _toy_families()
    assert ts.classify_transfer(1, 1, fams) == "repeat"
    assert ts.classify_transfer(1, 3, fams) == "same-family"
    assert ts.classify_transfer(1, 2, fams) == "near-transfer"
    assert ts.classify_transfer(1, 99, fams) == "unknown"
    reused = {"assignment": {"novel": False},
              "transfer": {"tag": "near-transfer"}}
    assert ts.is_novel_transfer(reused) is False
    same_family = {"assignment": {"novel": True},
                   "transfer": {"tag": "same-family"}}
    assert ts.is_novel_transfer(same_family) is False


def test_stage3_grind_bandit_16_deterministic_no_leak_split():
    """The family-held-out split is deterministic, covers every
    problem exactly once, and shares no family between train and
    test; the problem-level split is deterministic and leak-free
    on ids."""
    bank = problems.load()
    fams = problems.load_families()
    first = ts.split_by_family(bank, fams, seed=7, test_fraction=0.25)
    second = ts.split_by_family(bank, fams, seed=7, test_fraction=0.25)
    assert first == second
    assert set(first["train"]) | set(first["test"]) == {p["id"] for p in bank}
    assert not set(first["train"]) & set(first["test"])
    # No primary family leaks: a held-out family never appears in
    # training as a primary template. Secondary tags can still
    # co-occur across the split, and the audit reports that.
    train_primary = {problems.primary_family(pid, fams)
                     for pid in first["train"]}
    test_primary = {problems.primary_family(pid, fams)
                    for pid in first["test"]}
    assert not train_primary & test_primary
    assert set(first["test_families"]) == test_primary
    assert first["test_families"] and first["train_families"]
    by_problem = ts.split_by_problem(bank, seed=7, test_fraction=0.25)
    assert by_problem == ts.split_by_problem(bank, seed=7,
                                              test_fraction=0.25)
    assert not set(by_problem["train"]) & set(by_problem["test"])
    assert (set(by_problem["train"]) | set(by_problem["test"])
            == {p["id"] for p in bank})


def test_stage3_grind_bandit_16_split_audit_reports_blocked():
    """The audit names what cannot be split: a one-problem family
    cannot be held out within itself, and a topic whose problems all
    sit in one family cannot do within-topic cross-family transfer."""
    bank = [_problem(1, ("alpha",)), _problem(2, ("alpha",)),
            _problem(3, ("beta",)), _problem(4, ("beta",)),
            _problem(5, ("beta",))]
    fams = {
        "version": "families-v1",
        "families": [
            {"id": "solo", "label": "Solo", "description": "One member.",
             "examples": [1]},
            {"id": "pair", "label": "Pair",
             "description": "Several members.", "examples": [2]},
        ],
        "assignments": [
            {"problem_id": 1, "families": ["solo"]},
            {"problem_id": 2, "families": ["pair"]},
            {"problem_id": 3, "families": ["pair"]},
            {"problem_id": 4, "families": ["pair"]},
            {"problem_id": 5, "families": ["pair"]},
        ],
    }
    audit = ts.audit_family_leakage(bank, fams, [1, 3], [2, 4, 5])
    assert audit["shared_families"] == ["pair"]
    kinds = set()
    for b in audit["blocked"]:
        if b["kind"] == "single_problem_family":
            kinds.add((b["kind"], b["family"]))
        else:
            kinds.add((b["kind"], b["topic"]))
    assert ("single_problem_family", "solo") in kinds
    assert ("single_family_topic", "beta") in kinds


def test_stage3_grind_bandit_16_real_bank_audit():
    """On the real bank the audit runs clean: the problem-level split
    leaks families by design (and the audit says so), while the
    one-member prefix-sums family is reported blocked."""
    bank = problems.load()
    fams = problems.load_families()
    split = ts.split_by_problem(bank, seed=7, test_fraction=0.25)
    audit = ts.audit_family_leakage(bank, fams, split["train"],
                                    split["test"])
    assert audit["shared_families"]
    blocked = {(b["kind"], b.get("family")) for b in audit["blocked"]}
    assert ("single_problem_family", "prefix-sums") in blocked
    held_out = ts.split_by_family(bank, fams, seed=7, test_fraction=0.25)
    clean = ts.audit_family_leakage(bank, fams, held_out["train"],
                                    held_out["test"])
    # No primary family leaks across the held-out split, while the
    # audit still reports the secondary-tag leakage honestly: a
    # multi-tag problem can carry a held-out family as its second
    # tag, so shared families are reported, not hidden.
    held_train_primary = {problems.primary_family(pid, fams)
                          for pid in held_out["train"]}
    held_test_primary = {problems.primary_family(pid, fams)
                         for pid in held_out["test"]}
    assert not held_train_primary & held_test_primary
    assert clean["shared_families"]


def test_stage3_grind_bandit_16_validate_rejects_bad_data():
    """Validation catches an unknown problem id, an unknown family
    id, a duplicate assignment row, and a problem with no families."""
    bank = problems.load()
    fams = problems.load_families()
    bad = copy.deepcopy(fams)
    bad["assignments"].append({"problem_id": 99999,
                               "families": ["two-pointers"]})
    assert any("99999" in e for e in problems.validate_families(bank, bad))
    bad = copy.deepcopy(fams)
    bad["assignments"][0] = {"problem_id": 1,
                             "families": ["not-a-family"]}
    assert any("not-a-family" in e
               for e in problems.validate_families(bank, bad))
    bad = copy.deepcopy(fams)
    bad["assignments"].append({"problem_id": 1,
                               "families": ["hashmap-counting"]})
    assert any("duplicate" in e.lower()
               for e in problems.validate_families(bank, bad))
    bad = copy.deepcopy(fams)
    bad["assignments"] = [a for a in bad["assignments"]
                          if a["problem_id"] != 1]
    assert any("1" in e for e in problems.validate_families(bank, bad))
    with pytest.raises(ValueError):
        problems.load_families(path="/nonexistent/families.json")
