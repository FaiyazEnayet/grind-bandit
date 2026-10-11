"""Tests for the dashboard/GIF generators in scripts/.

The generators must stay wired to the real engine: these tests pin that the
demo state, the recommendations, and the animation frames all come from
grindbandit modules, not stub data.
"""

import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402
import make_dashboard  # noqa: E402
import make_demo_gif  # noqa: E402
from grindbandit import bandit, mastery, problems  # noqa: E402


def test_demo_state_matches_engine():
    state = common.build_demo_state(seed=42)
    assert len(state["log"]) == 30
    assert len(state["topics"]) > 0
    mean, _ = bandit.get_posterior(state)
    assert any(abs(v) > 1e-9 for v in mean)  # the bandit actually learned


def test_demo_state_reproducible_ignoring_wallclock():
    a = common.build_demo_state(seed=7)
    b = common.build_demo_state(seed=7)
    strip = lambda log: [{k: v for k, v in e.items() if k != "ts"} for e in log]
    assert strip(a["log"]) == strip(b["log"])


def test_recommend_matches_cli_pick():
    state = common.build_demo_state(seed=42)
    p, why, _ = common.recommend(state, seed=3)
    assert p["id"] == 994  # same pick the CLI prints for this demo state
    assert any("weakest topic" in line for line in why)


def test_log_attempt_reward_bounds():
    bank = problems.load()
    state = {"topics": {}, "problems": {}, "log": [], "bandit": {}}
    p = problems.by_id(bank, 1)
    now = time.time()
    reward, weak, after = common.log_attempt(state, p, True, 25, now)
    assert -1.0 <= reward <= 1.0
    assert reward > 0
    assert weak in p["topics"]
    assert after == mastery.mean(state, weak, now)


def test_dashboard_html_is_self_contained():
    html_text = make_dashboard.build()
    assert "<!DOCTYPE html>" in html_text
    assert "Rotting Oranges" in html_text  # the seed-3 Thompson pick
    assert "dp" in html_text and "two-pointers" in html_text
    # no external script or stylesheet: single file, works offline
    assert not re.search(r'<script\s+src="http', html_text)
    assert not re.search(r'<link[^>]+href="http', html_text)


def test_dashboard_html_covers_all_views():
    html_text = make_dashboard.build()
    for heading in ["What should I solve next", "Mastery across topics",
                    "Forgetting, visualized", "What the bandit learns from one log",
                    "The problem bank"]:
        assert heading in html_text


def test_gif_snapshots_come_from_the_engine():
    snaps = make_demo_gif.replay_snapshots(seed=42)
    assert len(snaps) == 30
    for label, vec in snaps:
        assert label.startswith("#")
        assert len(vec) == 27
        assert all(0.0 <= v <= 1.0 for v in vec)
    # mastery actually moved across the replay
    first = snaps[0][1]
    last = snaps[-1][1]
    assert any(abs(a - b) > 0.05 for a, b in zip(first, last))
