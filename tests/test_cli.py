"""CLI roundtrips. State is isolated per test via GRIND_BANDIT_STATE."""

import csv
import json

import pytest

from grindbandit import cli, problems, state as st


@pytest.fixture
def iso_state(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    monkeypatch.setenv("GRIND_BANDIT_STATE", str(p))
    return p


def test_log_then_next_roundtrip(iso_state, capsys):
    cli.main(["log", "1", "solved", "--minutes", "20"])
    out = capsys.readouterr().out
    assert "logged #1 Two Sum: solved" in out
    cli.main(["next", "--seed", "0"])
    out = capsys.readouterr().out
    assert out.startswith("#")
    assert "leetcode.com/problems/" in out
    assert "Why this one:" in out


def test_log_unknown_problem_fails(iso_state, capsys):
    with pytest.raises(SystemExit):
        cli.main(["log", "99999", "solved"])


def test_next_with_no_history(iso_state, capsys):
    cli.main(["next", "--seed", "0"])
    out = capsys.readouterr().out
    assert "Why this one:" in out


def test_next_is_deterministic_with_seed(iso_state, capsys):
    cli.main(["next", "--seed", "7"])
    first = capsys.readouterr().out
    cli.main(["next", "--seed", "7"])
    second = capsys.readouterr().out
    assert first == second


def test_all_solved_falls_back_to_review_mode(iso_state, capsys, monkeypatch):
    bank = problems.load()
    s = st.load()
    for p in bank:
        s.setdefault("problems", {})[str(p["id"])] = {
            "attempts": 1, "solved": 1, "last_ts": 0.0, "ever_solved": True,
        }
    st.save(s)
    cli.main(["next", "--seed", "0"])
    out = capsys.readouterr().out
    assert "Review mode" in out


def test_stats_runs(iso_state, capsys):
    cli.main(["log", "1", "solved"])
    cli.main(["log", "322", "failed"])
    cli.main(["stats"])
    out = capsys.readouterr().out
    assert "Mastery heatmap" in out
    assert "array" in out and "dp" in out


def test_review_with_no_log(iso_state, capsys):
    cli.main(["review"])
    assert "nothing logged yet" in capsys.readouterr().out


def test_review_shows_weeks(iso_state, capsys, monkeypatch):
    import grindbandit.cli as c
    base = 1_700_000_000.0
    monkeypatch.setattr(c, "_now", lambda: base)
    cli.main(["log", "1", "solved", "--minutes", "20"])
    monkeypatch.setattr(c, "_now", lambda: base + 8 * 86400.0)
    cli.main(["log", "322", "failed", "--minutes", "45"])
    cli.main(["review", "--weeks", "2"])
    out = capsys.readouterr().out
    assert "Weekly review" in out
    assert "attempts" in out


def test_export_csv_and_json(iso_state, tmp_path, capsys):
    cli.main(["log", "1", "solved", "--minutes", "20"])
    csv_out = tmp_path / "hist.csv"
    cli.main(["export", "--format", "csv", "--out", str(csv_out)])
    rows = list(csv.DictReader(open(csv_out)))
    assert len(rows) == 1
    assert rows[0]["problem_id"] == "1" and rows[0]["solved"] == "True"
    json_out = tmp_path / "state.json"
    cli.main(["export", "--format", "json", "--out", str(json_out)])
    data = json.load(open(json_out))
    assert len(data["log"]) == 1


def test_reset_wipes_state(iso_state, capsys):
    cli.main(["log", "1", "solved"])
    cli.main(["reset", "--yes"])
    s = st.load()
    assert s["log"] == [] and s["topics"] == {}


def test_slug_preserves_hyphens():
    assert problems.slug("K-diff Pairs in an Array") == "k-diff-pairs-in-an-array"
    assert problems.slug("Maximum Depth of N-ary Tree") == "maximum-depth-of-n-ary-tree"
    assert problems.slug("Two Sum IV - Input is a BST") == "two-sum-iv-input-is-a-bst"
    assert problems.slug("Reverse Nodes in k-Group") == "reverse-nodes-in-k-group"
    assert problems.slug("Pow(x, n)") == "powx-n"
    assert problems.slug("Pascal's Triangle") == "pascals-triangle"


def test_bank_problems_have_valid_shape():
    bank = problems.load()
    assert len(bank) >= 200
    ids = [p["id"] for p in bank]
    assert len(set(ids)) == len(ids)
    for p in bank:
        assert set(p) == {"id", "title", "difficulty", "topics"}
        assert p["difficulty"] in ("Easy", "Medium", "Hard")
        assert p["topics"]
        assert problems.url(p).startswith("https://leetcode.com/problems/")
        assert " " not in problems.url(p)
