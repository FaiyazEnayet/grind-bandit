"""Stage 3 GB7: state migration and model documentation.

Before this change, state files written by older versions had no
schema version, and loads only migrated them in memory: nothing ever
wrote the migrated form back, and no backup of the original was kept.
The reward model math lived only in module docstrings, with no
worked numerical examples a reader could check against the code.

These tests pin the GB7 behavior: a versioned schema with a
documented upgrade register, a persisted migration path that backs
up the original file before writing, migration that invents no
history, and the worked examples from docs/model.md reproduced
exactly by the engine.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from grindbandit import bandit, cli, mastery  # noqa: E402
from grindbandit import state as st  # noqa: E402

T0 = 1_700_000_000.0
DAY = 86400.0
FIXTURE = Path(__file__).parent / "fixtures" / "state-v0-example.json"


def _blank_state():
    return {"topics": {}, "problems": {}, "log": [], "bandit": {},
            "recommendations": [], "delayed_quizzes": []}


def _problem(pid=1, topic="array"):
    return {"id": pid, "title": f"P{pid}", "difficulty": "Easy",
            "topics": [topic]}


def _fixture():
    return json.loads(FIXTURE.read_text())


# --- versioned schema ------------------------------------------------------


def test_stage3_grind_bandit_7_schema_version_is_stamped_on_new_state():
    assert st.SCHEMA_VERSION == 1
    blank = st._blank_state()
    assert blank["schema_version"] == 1
    assert st.state_version(blank) == 1
    assert st.state_version({}) == 0


def test_stage3_grind_bandit_7_v0_fixture_migrates_deterministically():
    old = _fixture()
    assert st.state_version(old) == 0
    first = st.migrate_state(old)
    second = st.migrate_state(old)
    assert first == second
    assert first["schema_version"] == 1
    assert first["recommendations"] == []
    assert first["delayed_quizzes"] == []
    assert first["topics"] == old["topics"]
    assert first["problems"] == old["problems"]
    assert first["log"] == old["log"]
    assert first["bandit"] == old["bandit"]
    assert old == _fixture()


def test_stage3_grind_bandit_7_migration_invents_no_history():
    old = _fixture()
    migrated = st.migrate_state(old)
    for entry in migrated["log"]:
        assert "attempt_id" not in entry
        assert "recommendation_id" not in entry
        assert "linkage_method" not in entry
    assert migrated["recommendations"] == []
    assert migrated["delayed_quizzes"] == []


def test_stage3_grind_bandit_7_load_leaves_v0_file_on_disk_untouched(tmp_path):
    path = tmp_path / "state.json"
    path.write_bytes(FIXTURE.read_bytes())
    loaded = st.load(str(path))
    assert loaded["schema_version"] == 1
    assert path.read_bytes() == FIXTURE.read_bytes()


def test_stage3_grind_bandit_7_load_rejects_newer_version(tmp_path):
    path = tmp_path / "state.json"
    newer = _blank_state()
    newer["schema_version"] = 99
    path.write_text(json.dumps(newer))
    with pytest.raises(ValueError, match="newer"):
        st.load(str(path))


# --- persisted migration with backup ---------------------------------------


def test_stage3_grind_bandit_7_migrate_file_backs_up_and_migrates(tmp_path):
    path = tmp_path / "state.json"
    path.write_bytes(FIXTURE.read_bytes())
    result = st.migrate_file(str(path))
    assert result["from_version"] == 0
    assert result["to_version"] == 1
    backup = Path(result["backup_path"])
    assert backup.exists()
    assert backup.read_bytes() == FIXTURE.read_bytes()
    migrated = json.loads(path.read_text())
    assert migrated["schema_version"] == 1
    assert migrated["recommendations"] == []
    assert migrated["delayed_quizzes"] == []
    assert migrated["log"] == _fixture()["log"]
    assert st.load(str(path)) == migrated


def test_stage3_grind_bandit_7_migrate_file_is_a_noop_when_current(tmp_path):
    path = tmp_path / "state.json"
    path.write_bytes(FIXTURE.read_bytes())
    first = st.migrate_file(str(path))
    after_first = path.read_bytes()
    second = st.migrate_file(str(path))
    assert second["from_version"] == 1
    assert second["backup_path"] is None
    assert path.read_bytes() == after_first
    assert Path(first["backup_path"]).read_bytes() == FIXTURE.read_bytes()


def test_stage3_grind_bandit_7_migrate_file_refuses_newer_version(tmp_path):
    path = tmp_path / "state.json"
    newer = _blank_state()
    newer["schema_version"] = 99
    path.write_text(json.dumps(newer))
    with pytest.raises(ValueError, match="newer"):
        st.migrate_file(str(path))
    assert json.loads(path.read_text()) == newer


def test_stage3_grind_bandit_7_migrate_file_missing_file_raises(tmp_path):
    path = tmp_path / "state.json"
    with pytest.raises(FileNotFoundError):
        st.migrate_file(str(path))


def test_stage3_grind_bandit_7_demo_state_is_v0_and_migrates():
    from common import build_demo_state

    demo = build_demo_state()
    assert st.state_version(demo) == 0
    first = st.migrate_state(demo)
    second = st.migrate_state(demo)
    assert first == second
    assert first["schema_version"] == 1
    assert len(first["log"]) == 30


def test_stage3_grind_bandit_7_cli_migrate_command(tmp_path, capsys):
    path = tmp_path / "state.json"
    path.write_bytes(FIXTURE.read_bytes())
    cli.main(["migrate", "--path", str(path)])
    out = capsys.readouterr().out
    assert "schema version 0" in out
    assert "schema version 1" in out
    assert "backup" in out
    assert json.loads(path.read_text())["schema_version"] == 1
    cli.main(["migrate", "--path", str(path)])
    out = capsys.readouterr().out
    assert "already at schema version 1" in out


def test_stage3_grind_bandit_7_cli_migrate_missing_file(tmp_path, capsys):
    cli.main(["migrate", "--path", str(tmp_path / "state.json")])
    assert "no state file" in capsys.readouterr().out


# --- worked model examples (docs/model.md) ---------------------------------


def test_stage3_grind_bandit_7_example_fresh_topic_mastery_is_one_half():
    state = _blank_state()
    assert mastery.mean(state, "array", T0) == 0.5


def test_stage3_grind_bandit_7_example_one_solve_is_two_thirds():
    state = _blank_state()
    mastery.update(state, ["array"], True, T0)
    assert state["topics"]["array"]["alpha"] == 2.0
    assert state["topics"]["array"]["beta"] == 1.0
    assert mastery.mean(state, "array", T0) == pytest.approx(2.0 / 3.0)


def test_stage3_grind_bandit_7_example_ten_solves_then_ninety_idle_days():
    state = _blank_state()
    for _ in range(10):
        mastery.update(state, ["array"], True, T0)
    assert mastery.mean(state, "array", T0 + 90 * DAY) == pytest.approx(
        0.6923076923, abs=1e-9
    )


def test_stage3_grind_bandit_7_example_failure_after_decay_lowers_mastery():
    state = _blank_state()
    for _ in range(10):
        mastery.update(state, ["array"], True, T0)
    mastery.update(state, ["array"], False, T0 + 90 * DAY)
    topic = state["topics"]["array"]
    assert topic["alpha"] == pytest.approx(2.25)
    assert topic["beta"] == 2.0
    assert mastery.mean(state, "array", T0 + 90 * DAY) == pytest.approx(
        0.5294117647, abs=1e-9
    )


def test_stage3_grind_bandit_7_example_reward_is_weakest_topic_mastery_change():
    for solved, expected in ((True, 1.0 / 6.0), (False, -1.0 / 6.0)):
        state = _blank_state()
        p = _problem(28, "string")
        tm = {t: mastery.mean(state, t, T0) for t in p["topics"]}
        weak = bandit.weakest_topic(p, tm)
        before = tm[weak]
        mastery.update(state, p["topics"], solved, T0)
        after = mastery.mean(state, weak, T0)
        reward = max(-1.0, min(1.0, after - before))
        assert reward == pytest.approx(expected)


def test_stage3_grind_bandit_7_example_fresh_context_vector():
    state = _blank_state()
    p = {"id": 28, "title": "Find the Index of the First Occurrence "
                            "in a String",
         "difficulty": "Easy", "topics": ["string"]}
    assert bandit.context(p, state, T0).tolist() == [0.5, 0.5, 0.5, 1.0, 1.0]


def test_stage3_grind_bandit_7_example_posterior_after_one_observation():
    import numpy as np

    state = _blank_state()
    x = np.array([0.5, 0.5, 0.5, 1.0, 1.0])
    bandit.observe(state, x, 1.0 / 6.0)
    assert state["bandit"]["mean"] == pytest.approx(
        [0.0292397661, 0.0292397661, 0.0292397661, 0.0584795322, 0.0584795322],
        abs=1e-9,
    )
    assert state["bandit"]["prec"] == [
        [3.5, 2.5, 2.5, 5.0, 5.0],
        [2.5, 3.5, 2.5, 5.0, 5.0],
        [2.5, 2.5, 3.5, 5.0, 5.0],
        [5.0, 5.0, 5.0, 11.0, 10.0],
        [5.0, 5.0, 5.0, 10.0, 11.0],
    ]


def test_stage3_grind_bandit_7_example_opposite_rewards_cancel():
    import numpy as np

    state = _blank_state()
    x = np.array([0.5, 0.5, 0.5, 1.0, 1.0])
    bandit.observe(state, x, 1.0 / 6.0)
    bandit.observe(state, x, -1.0 / 6.0)
    assert state["bandit"]["mean"] == pytest.approx([0.0] * 5, abs=1e-12)


def test_stage3_grind_bandit_7_example_thompson_draw_is_reproducible():
    state = _blank_state()
    theta = bandit.sample_theta(state, seed=7)
    assert theta.tolist() == pytest.approx(
        [0.0012301534, 0.2987455375, -0.2741378554, -0.8905918388,
         -0.4546707852],
        abs=1e-9,
    )
    assert bandit.sample_theta(state, seed=7).tolist() == theta.tolist()


def test_stage3_grind_bandit_7_example_review_interval_classes():
    bank = [_problem(1, "array")]
    for age_days, cls, eligible in ((3.0, "review_cooldown", False),
                                    (8.0, "due_review", True)):
        state = _blank_state()
        state["problems"] = {"1": {"attempts": 1, "solved": 1,
                                   "last_ts": T0 - age_days * DAY,
                                   "ever_solved": True}}
        entry = cli.assess_eligibility(bank, state, T0)[0]
        assert entry["class"] == cls
        assert entry["eligible"] is eligible
