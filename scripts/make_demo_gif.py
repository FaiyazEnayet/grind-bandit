"""Build docs/demo.gif: an animated tour of the dashboard's mastery view.

Replays the scripted 30-attempt demo history through the real grindbandit
engine and animates the per-topic mastery bars after every attempt. The
frames are real engine output, not mock data.

Usage:
    pip install matplotlib pillow   # only needed to (re)generate
    python scripts/make_demo_gif.py

Output: docs/demo.gif
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import common  # noqa: E402
from common import BANK, DAY, TOPICS, log_attempt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from grindbandit import mastery  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

OUT = Path(__file__).parent.parent / "docs" / "demo.gif"


def replay_snapshots(seed: int = 42):
    """Run the demo loop, snapshotting (label, mastery vector) after each attempt."""
    state = {"topics": {}, "problems": {}, "log": [], "bandit": {}}
    rng = random.Random(seed)
    start = time.time() - 13 * DAY
    strong = {"array", "string"}
    weak = {"dp", "graph", "backtracking"}
    order = BANK[:]
    rng.shuffle(order)
    snaps = []
    for n, p in enumerate(order[:30]):
        topics = set(p["topics"])
        rate = 0.9 if topics & strong else (0.3 if topics & weak else 0.7)
        solved = rng.random() < rate
        ts = start + n * (13 * DAY / 30)
        log_attempt(state, p, solved, rng.randint(15, 60), ts)
        vec = [mastery.mean(state, t, ts) for t in TOPICS]
        snaps.append((f"#{p['id']} {p['title']} ({'solved' if solved else 'failed'})",
                      vec))
    return snaps


def color(v: float) -> str:
    return "#d73027" if v < 0.4 else "#fee08b" if v < 0.65 else "#1a9850"


def main() -> None:
    snaps = replay_snapshots()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("white")

    def draw(i: int):
        ax.clear()
        label, vec = snaps[i]
        order = sorted(range(len(TOPICS)), key=lambda j: vec[j])
        topics = [TOPICS[j] for j in order]
        vals = [vec[j] for j in order]
        ax.barh(topics, vals, color=[color(v) for v in vals], edgecolor="white")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("mastery")
        ax.set_title(f"grind-bandit: mastery after attempt {i + 1}/30\n{label}",
                     fontsize=11, loc="left")
        weakest = topics[0]
        ax.text(0.98, 0.02, f"weakest: {weakest} ({vals[0]:.0%})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, bbox=dict(facecolor="white", edgecolor="#ddd"))
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(snaps), interval=600, repeat_delay=2000)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(OUT), writer="pillow", fps=2)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(snaps)} frames)")


if __name__ == "__main__":
    main()
