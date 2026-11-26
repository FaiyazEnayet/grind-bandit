"""Stage 3 GB11: multi-topic credit-assignment sensitivity.

Gate tests for the offline-only credit audit. mastery.update credits every
tagged topic equally, but 190 of the 218 curated problems carry multiple
tags, so the choice of credit rule moves the mastery picture. The audit
replays the same logged attempts under three rules (full, fractional,
user-reported blocked topic) and reports the per-topic differences.

Nothing here is a learning improvement: the audit is counterfactual
modeling of one log under different assumptions, it never touches live
state, and a user-reported blocked topic is a modeling input, never
ground truth.
"""

import copy
import json
import math
from pathlib import Path

import pytest

from grindbandit import credit_audit, mastery

DAY = 86400.0
T0 = 1_750_000_000.0


def _entry(ts, topics, solved, **extra):
    e = {"ts": ts, "topics": list(topics), "solved": solved}
    e.update(extra)
    return e


def test_stage3_grind_bandit_11_single_topic_rules_agree():
    solved = [_entry(T0, ["array"], True, blocked_topic="array")]
    failed = [_entry(T0, ["array"], False, blocked_topic="array")]
    for entries, expected in ((solved, 2 / 3), (failed, 1 / 3)):
        out = credit_audit.compare_rules(entries, T0)
        for rule in credit_audit.RULES:
            assert out["rules"][rule]["mastery"]["array"] == pytest.approx(expected)
        assert out["differences"]["array"]["max_spread"] == pytest.approx(0.0)


def test_stage3_grind_bandit_11_multi_topic_solved_known_answers():
    entries = [_entry(T0, ["array", "hash-table"], True,
                      blocked_topic="array")]
    out = credit_audit.compare_rules(entries, T0)
    full = out["rules"]["full"]["mastery"]
    assert full["array"] == pytest.approx(2 / 3)
    assert full["hash-table"] == pytest.approx(2 / 3)
    assert out["rules"]["full"]["counts"]["array"]["alpha"] == 2.0
    frac = out["rules"]["fractional"]["mastery"]
    assert frac["array"] == pytest.approx(0.6)
    assert frac["hash-table"] == pytest.approx(0.6)
    assert out["rules"]["fractional"]["counts"]["array"]["alpha"] == 1.5
    blocked = out["rules"]["blocked"]["mastery"]
    assert blocked["array"] == pytest.approx(2 / 3)
    assert blocked["hash-table"] == pytest.approx(0.5)
    assert out["rules"]["blocked"]["counts"]["hash-table"]["attempts"] == 0
    assert out["differences"]["hash-table"]["max_spread"] == pytest.approx(2 / 3 - 0.5)


def test_stage3_grind_bandit_11_multi_topic_failed_known_answers():
    entries = [_entry(T0, ["array", "hash-table"], False,
                      blocked_topic="array")]
    out = credit_audit.compare_rules(entries, T0)
    assert out["rules"]["full"]["mastery"]["array"] == pytest.approx(1 / 3)
    assert out["rules"]["fractional"]["mastery"]["array"] == pytest.approx(0.4)
    assert out["rules"]["blocked"]["mastery"]["array"] == pytest.approx(1 / 3)
    assert out["rules"]["blocked"]["mastery"]["hash-table"] == pytest.approx(0.5)


def test_stage3_grind_bandit_11_full_matches_production_replay():
    entries = [
        _entry(T0, ["array", "hash-table"], True),
        _entry(T0 + DAY, ["array"], False),
        _entry(T0 + 40 * DAY, ["dp", "array"], True),
        _entry(T0 + 41 * DAY, ["dp"], False),
        _entry(T0 + 90 * DAY, ["graph"], True),  # past upto_ts: ignored
    ]
    upto = T0 + 45 * DAY
    audit = credit_audit.replay_under_rule(entries, upto, "full",
                                           half_life_days=30.0)
    live = mastery.replay(entries, upto, half_life_days=30.0)
    assert audit["mastery"] == live
    assert audit["meta"]["n_attempts"] == 4


def test_stage3_grind_bandit_11_stable_replay():
    entries = [
        _entry(T0, ["array", "hash-table"], True, blocked_topic="hash-table"),
        _entry(T0 + DAY, ["dp", "array"], False),
    ]
    first = credit_audit.compare_rules(entries, T0 + 2 * DAY)
    second = credit_audit.compare_rules(entries, T0 + 2 * DAY)
    assert first == second


def test_stage3_grind_bandit_11_no_mutation_of_entries():
    entries = [
        _entry(T0, ["array", "hash-table"], True, blocked_topic="array"),
        _entry(T0 + DAY, ["dp"], False),
    ]
    before = copy.deepcopy(entries)
    credit_audit.compare_rules(entries, T0 + 2 * DAY)
    assert entries == before
    assert len(entries) == 2


def test_stage3_grind_bandit_11_rejects_invalid_tag_input():
    bad = [
        _entry(T0, [], True),
        {"ts": T0, "topics": "array", "solved": True},
        _entry(T0, ["array", 7], True),
        _entry(T0, ["array", ""], True),
        {"ts": T0, "topics": ["array"]},
        _entry(T0, ["array"], "yes"),
        _entry(T0, ["array"], True, blocked_topic="dp"),
        _entry(T0, ["array"], True, blocked_topic=3),
        {"topics": ["array"], "solved": True},
        _entry(float("nan"), ["array"], True),
        _entry(T0, ["array"], True, blocked_topic="array"),
    ]
    # The last entry is valid on its own; everything before it must fail.
    for entry in bad[:-1]:
        with pytest.raises(ValueError):
            credit_audit.compare_rules([entry], T0)
    # And the valid one passes.
    credit_audit.compare_rules([bad[-1]], T0)


def test_stage3_grind_bandit_11_blocked_fallback_is_counted():
    entries = [
        _entry(T0, ["array", "hash-table"], True, blocked_topic="array"),
        _entry(T0 + DAY, ["array", "dp"], False),  # no blocked topic
        _entry(T0 + 2 * DAY, ["dp"], True),  # single topic, no report
    ]
    out = credit_audit.compare_rules(entries, T0 + 2 * DAY)
    blocked = out["rules"]["blocked"]
    assert blocked["meta"]["blocked_fallbacks"] == 2
    assert out["rules"]["full"]["meta"]["blocked_fallbacks"] == 0
    assert out["rules"]["fractional"]["meta"]["blocked_fallbacks"] == 0
    # The fallback attempt is credited fractionally under the blocked rule.
    # The solve a day later persists one day of decay on that half-credit
    # first (half-life 30 days), so the stored beta is 1 + 0.5 * 0.5**(1/30).
    assert blocked["counts"]["dp"]["beta"] == pytest.approx(
        1 + 0.5 * 0.5 ** (1 / 30))


def test_stage3_grind_bandit_11_results_labeled_counterfactual():
    entries = [_entry(T0, ["array", "hash-table"], True,
                      blocked_topic="array")]
    out = credit_audit.compare_rules(entries, T0)
    assert "counterfactual" in out["label"]
    assert "not a learning improvement" in out["label"]
    for rule in credit_audit.RULES:
        assert out["rules"][rule]["assumption"]
    assert "subjective" in out["rules"]["blocked"]["assumption"]
    assert "ground truth" in out["rules"]["blocked"]["assumption"]
    assert out["max_spread_topic"] == "hash-table"
    assert out["max_spread"] == pytest.approx(2 / 3 - 0.5)


def test_stage3_grind_bandit_11_real_bank_multi_tag_share():
    bank_path = Path(__file__).parent.parent / "src" / "grindbandit" / "data" / "problems.json"
    bank = json.loads(bank_path.read_text())
    multi = sum(1 for p in bank if len(set(p["topics"])) > 1)
    assert len(bank) == 218
    assert multi == 190
    entries = [
        _entry(T0 + i * DAY, p["topics"], i % 2 == 0,
               **({"blocked_topic": p["topics"][0]} if i % 3 == 0 else {}))
        for i, p in enumerate(bank[:6])
    ]
    out = credit_audit.compare_rules(entries, T0 + 6 * DAY)
    assert out["n_attempts"] == 6
    assert out["n_multi_tag_attempts"] >= 5
    assert out["differences"]


def test_stage3_grind_bandit_11_decay_matches_production():
    entries = [_entry(T0, ["array", "hash-table"], True)]
    upto = T0 + 90 * DAY
    audit = credit_audit.replay_under_rule(entries, upto, "full",
                                           half_life_days=30.0)
    live = mastery.replay(entries, upto, half_life_days=30.0)
    assert audit["mastery"] == live
    # After 90 days (three half-lives) the single win has mostly decayed.
    assert audit["mastery"]["array"] == pytest.approx(
        (1 + 0.125) / (2 + 0.125), rel=1e-9)


def test_stage3_grind_bandit_11_empty_log_reports_nothing():
    out = credit_audit.compare_rules([], T0)
    assert out["rules"]["full"]["mastery"] == {}
    assert out["differences"] == {}
    assert out["max_spread_topic"] is None
    assert out["max_spread"] == 0.0
    assert math.isfinite(out["max_spread"])
