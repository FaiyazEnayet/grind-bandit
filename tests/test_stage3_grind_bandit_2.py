"""Stage 3 GB2: persistence hardening.

Before this change the state file was written with a plain open/write,
so an interrupted serialization could leave a partial file behind and
the next load failed. Loads did no schema validation, so a state file
with the wrong shape was only found when some later lookup broke. And
two CLI processes could save over each other with no conflict signal.

These tests pin the hardened behavior: basename paths work, an
interrupted save leaves the prior file readable, a contended save
fails with an explicit conflict instead of silently overwriting, loads
validate the schema, and older state files migrate deterministically.
"""

import fcntl
import json

import pytest

from grindbandit import state as st

T0 = 1_700_000_000.0


def _blank_state():
    return {"topics": {}, "problems": {}, "log": [], "bandit": {},
            "recommendations": [], "delayed_quizzes": []}


def _problem(pid=1, topic="array"):
    return {"id": pid, "title": f"P{pid}", "difficulty": "Easy",
            "topics": [topic]}


# --- basename paths --------------------------------------------------------


def test_stage3_grind_bandit_2_basename_path_round_trips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = st.save(_blank_state(), "state.json")
    assert saved == "state.json"
    assert st.load("state.json") == _blank_state()


# --- interrupted serialization ---------------------------------------------


def test_stage3_grind_bandit_2_interrupted_save_leaves_prior_file_readable(
        tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    good = _blank_state()
    st.record_attempt(good, _problem(1), True, 20, 0.1, T0)
    st.save(good, str(path))

    def _crash_mid_write(obj, f, *args, **kwargs):
        f.write('{"topics": ')
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(st.json, "dump", _crash_mid_write)
    with pytest.raises(RuntimeError, match="simulated crash"):
        st.save(good, str(path))

    assert st.load(str(path)) == good
    assert list(tmp_path.glob(".state-*.tmp")) == []


# --- concurrent CLI protection ---------------------------------------------


def test_stage3_grind_bandit_2_contended_save_raises_explicit_conflict(tmp_path):
    path = tmp_path / "state.json"
    st.save(_blank_state(), str(path))
    lock_path = str(path) + ".lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        with pytest.raises(st.StateConflictError):
            st.save(_blank_state(), str(path))
    assert st.load(str(path)) == _blank_state()


def test_stage3_grind_bandit_2_sequential_saves_retain_both_attempts(tmp_path):
    path = tmp_path / "state.json"
    first = _blank_state()
    st.record_attempt(first, _problem(1), True, 20, 0.1, T0)
    st.save(first, str(path))
    second = st.load(str(path))
    st.record_attempt(second, _problem(2), False, 35, -0.2, T0 + 60.0)
    st.save(second, str(path))
    reloaded = st.load(str(path))
    assert [e["problem_id"] for e in reloaded["log"]] == [1, 2]


# --- schema validation and deterministic migration -------------------------


def test_stage3_grind_bandit_2_load_rejects_wrong_shaped_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"topics": {}, "log": "not-a-list"}))
    with pytest.raises(ValueError, match="log"):
        st.load(str(path))


def test_stage3_grind_bandit_2_load_rejects_non_dict_entries(tmp_path):
    path = tmp_path / "state.json"
    bad = _blank_state()
    bad["log"] = ["not-an-entry"]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="log"):
        st.load(str(path))


def test_stage3_grind_bandit_2_older_state_migrates_deterministically(tmp_path):
    path = tmp_path / "state.json"
    old = {"topics": {"array": {"attempts": 1}}, "problems": {},
           "log": [], "bandit": {}}
    path.write_text(json.dumps(old))
    first = st.load(str(path))
    second = st.load(str(path))
    assert first == second
    assert first["recommendations"] == []
    assert first["delayed_quizzes"] == []
    assert first["topics"] == {"array": {"attempts": 1}}
    assert json.loads(path.read_text()) == old


def test_stage3_grind_bandit_2_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"topics": ')
    with pytest.raises(ValueError):
        st.load(str(path))


def test_stage3_grind_bandit_2_save_validates_before_writing(tmp_path):
    path = tmp_path / "state.json"
    st.save(_blank_state(), str(path))
    bad = _blank_state()
    bad["recommendations"] = "not-a-list"
    with pytest.raises(ValueError, match="recommendations"):
        st.save(bad, str(path))
    assert st.load(str(path)) == _blank_state()
