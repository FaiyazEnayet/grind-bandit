"""Build docs/dashboard.html: a single self-contained interactive dashboard.

Every number on the page is computed by the real grindbandit engine
(src/grindbandit). Nothing is stubbed or reimplemented.

Usage:
    pip install plotly        # only needed to (re)generate
    python scripts/make_dashboard.py

Output: docs/dashboard.html — one file, Plotly.js embedded, ready for
GitHub Pages. No Python needed to view it.
"""

from __future__ import annotations

import copy
import html
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import common  # noqa: E402
from common import BANK, DAY, TOPICS, build_demo_state, log_attempt, recommend  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from grindbandit import mastery, problems  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

OUT = Path(__file__).parent.parent / "docs" / "dashboard.html"


def mastery_rows(state: dict, now: float) -> list[tuple[str, float, int, float, float | None]]:
    rows = []
    for t in TOPICS:
        info = state.get("topics", {}).get(t, {})
        attempts = info.get("attempts", 0)
        solved = info.get("solved", 0)
        rows.append((
            t,
            mastery.mean(state, t, now),
            attempts,
            (solved / attempts) if attempts else 0.0,
            info.get("last_ts"),
        ))
    return sorted(rows, key=lambda r: r[1])


def describe_last(last_ts, now: float) -> str:
    if last_ts is None:
        return "never"
    d = (now - last_ts) / DAY
    return "today" if d < 1 else f"{int(d)}d ago"


def fig_mastery_bars(rows) -> go.Figure:
    topics = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#d73027" if v < 0.4 else "#fee08b" if v < 0.65 else "#1a9850" for v in vals]
    fig = go.Figure(go.Bar(
        x=topics, y=vals, marker_color=colors,
        text=[f"{v:.0%}" for v in vals], textposition="outside",
        hovertemplate="%{x}<br>mastery %{y:.0%}<extra></extra>",
    ))
    fig.update_layout(
        title="Per-topic mastery, weakest to strongest",
        xaxis_title="topic", yaxis_title="mastery", yaxis_range=[0, 1.05],
        height=420, margin=dict(b=140),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def fig_forgetting(state: dict, now: float, rows) -> go.Figure:
    days = list(range(0, 181, 5))
    fig = go.Figure()
    for t in TOPICS:
        ys = [mastery.mean(state, t, now + d * DAY) for d in days]
        fig.add_trace(go.Scatter(
            x=days, y=ys, mode="lines", name=t, visible=False,
            hovertemplate=f"{t}<br>day %{{x}}: %{{y:.0%}}<extra></extra>",
        ))
    weakest = rows[0][0]
    for tr in fig.data:
        if tr.name == weakest:
            tr.visible = True
    buttons = [
        dict(label=t, method="update",
             args=[{"visible": [name == t for name in TOPICS]},
                   {"title": f"Forgetting projection: {t}"}])
        for t in TOPICS
    ]
    fig.update_layout(
        title=f"Forgetting projection: {weakest}",
        xaxis_title="days from now", yaxis_title="projected mastery",
        yaxis_range=[0.4, 1.02], height=420,
        updatemenus=[dict(active=TOPICS.index(weakest), buttons=buttons,
                          x=0.0, y=1.12, xanchor="left", yanchor="top")],
        annotations=[dict(text="Topic:", x=0, y=1.1, xref="paper", yref="paper",
                          showarrow=False, xanchor="right")],
    )
    return fig


def fig_log_effect(state: dict, now: float):
    """Mastery of the recommended problem's topics: before vs after a solve/fail."""
    p, _, _ = recommend(state, seed=3, now=now)
    topics = p["topics"]
    before = {t: mastery.mean(state, t, now) for t in topics}

    s_state = copy.deepcopy(state)
    reward_s, weak, _ = log_attempt(s_state, p, True, 30, now)
    after_s = {t: mastery.mean(s_state, t, now) for t in topics}

    f_state = copy.deepcopy(state)
    reward_f, _, _ = log_attempt(f_state, p, False, 45, now)
    after_f = {t: mastery.mean(f_state, t, now) for t in topics}

    fig = go.Figure()
    fig.add_trace(go.Bar(x=topics, y=[before[t] for t in topics], name="before",
                         marker_color="#9e9e9e"))
    fig.add_trace(go.Bar(x=topics, y=[after_s[t] for t in topics], name="after solve",
                         marker_color="#1a9850", visible=True))
    fig.add_trace(go.Bar(x=topics, y=[after_f[t] for t in topics], name="after fail",
                         marker_color="#d73027", visible=False))
    fig.update_layout(
        title=f"One log moves mastery: #{p['id']} {p['title']}",
        barmode="group", yaxis_title="mastery", yaxis_range=[0, 1.05], height=400,
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.15,
                          xanchor="left", yanchor="top",
                          buttons=[
                              dict(label="solved", method="update",
                                   args=[{"visible": [True, True, False]}]),
                              dict(label="failed", method="update",
                                   args=[{"visible": [True, False, True]}]),
                          ])],
    )
    note = (f"Reward is the mastery gain in the weakest topic ({weak}): "
            f"{reward_s:+.3f} for a solve, {reward_f:+.3f} for a fail. "
            f"The bandit observes this reward and updates its posterior.")
    return fig, note, p


def fig_difficulty() -> go.Figure:
    diffs = Counter(p["difficulty"] for p in BANK)
    order = ["Easy", "Medium", "Hard"]
    fig = go.Figure(go.Bar(
        x=order, y=[diffs[o] for o in order],
        marker_color=["#1a9850", "#fee08b", "#d73027"],
        text=[diffs[o] for o in order], textposition="outside",
    ))
    fig.update_layout(title="Problem bank by difficulty", yaxis_title="problems",
                      height=340)
    return fig


def bank_table() -> go.Figure:
    fig = go.Figure(go.Table(
        header=dict(values=["id", "title", "difficulty", "topics"],
                    fill_color="#eeeeee", align="left"),
        cells=dict(values=[
            [p["id"] for p in BANK],
            [p["title"] for p in BANK],
            [p["difficulty"] for p in BANK],
            [", ".join(p["topics"]) for p in BANK],
        ], align="left"),
    ))
    fig.update_layout(title="The full problem bank (218 problems)", height=600)
    return fig


def recommendation_cards(state: dict, now: float, seeds=range(10)) -> str:
    """Precomputed Thompson-sampling picks, one card per seed, switched by JS."""
    cards = []
    options = []
    for i, seed in enumerate(seeds):
        p, why, review_mode = recommend(state, seed=seed, now=now)
        weak_line = why[0]
        card = f"""
        <div class="rec-card" id="rec-{seed}" style="display:{'block' if i == 0 else 'none'}">
          <h3>#{p['id']} {html.escape(p['title'])} ({p['difficulty']})</h3>
          <p><a href="{problems.url(p)}" target="_blank" rel="noopener">
             leetcode.com/problems/{problems.slug(p['title'])}</a><br/>
             Topics: {html.escape(', '.join(p['topics']))}</p>
          <p><b>Why this one:</b></p>
          <ul>{''.join(f'<li>{html.escape(line)}</li>' for line in why)}</ul>
          {('<p><i>Review mode: every problem in the bank is solved.</i></p>'
            if review_mode else '')}
        </div>"""
        cards.append(card)
        options.append(f'<option value="{seed}"{" selected" if i == 0 else ""}>seed {seed}</option>')
    js = """
    <script>
    function showRec(seed) {
      document.querySelectorAll('.rec-card').forEach(el => el.style.display = 'none');
      document.getElementById('rec-' + seed).style.display = 'block';
    }
    </script>"""
    return (f'<label><b>Seed</b> <select onchange="showRec(this.value)">'
            f'{"".join(options)}</select></label>'
            f'<p>Same seed gives the same pick. Switch seeds to watch Thompson '
            f'sampling explore: early on the posterior is wide, so different '
            f'samples land on different problems.</p>'
            f'{"".join(cards)}{js}')


def build() -> str:
    state = build_demo_state(seed=42)
    now = time.time()
    rows = mastery_rows(state, now)
    n_log = len(state["log"])
    solved = sum(1 for e in state["log"] if e["solved"])
    minutes = sum(e["minutes"] for e in state["log"])
    diffs = Counter(p["difficulty"] for p in BANK)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fig_effect, effect_note, rec_problem = fig_log_effect(state, now)

    figs = {
        "mastery": fig_mastery_bars(rows),
        "forgetting": fig_forgetting(state, now, rows),
        "effect": fig_effect,
        "difficulty": fig_difficulty(),
        "bank": bank_table(),
    }
    divs = {k: f.to_html(full_html=False, include_plotlyjs=False) for k, f in figs.items()}

    import plotly.offline as po

    plotly_js = po.get_plotlyjs()  # bundle it: the file works fully offline

    weakest5 = ", ".join(f"{t} ({m:.0%})" for t, m, *_ in rows[:5])
    strongest5 = ", ".join(f"{t} ({m:.0%})" for t, m, *_ in rows[-5:])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>grind-bandit dashboard</title>
<script>{plotly_js}</script>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 24px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0; }} .sub {{ color: #555; margin-top: 4px; }}
  .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 20px 0; }}
  .stat {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; }}
  .stat b {{ font-size: 1.4em; display: block; }}
  section {{ margin: 40px 0; }} h2 {{ border-bottom: 2px solid #eee; padding-bottom: 6px; }}
  .rec-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-top: 12px; }}
  footer {{ color: #777; font-size: 0.85em; margin-top: 48px; border-top: 1px solid #eee; padding-top: 12px; }}
</style>
</head>
<body>
<h1>🎯 grind-bandit dashboard</h1>
<p class="sub">Your LeetCode grind as a contextual bandit. Every number below was
computed by the real engine in <code>src/grindbandit</code> from a scripted
30-attempt demo history (seed 42). No stubs.</p>

<div class="stats">
  <div class="stat"><b>{len(BANK)}</b>problems in bank</div>
  <div class="stat"><b>{len(TOPICS)}</b>topics</div>
  <div class="stat"><b>{diffs['Easy']}/{diffs['Medium']}/{diffs['Hard']}</b>easy / medium / hard</div>
  <div class="stat"><b>{n_log}</b>demo attempts</div>
  <div class="stat"><b>{solved / n_log:.0%}</b>solve rate</div>
  <div class="stat"><b>{minutes}</b>minutes logged</div>
</div>

<section>
<h2>What should I solve next</h2>
<p>One theta is sampled from the bandit's Gaussian posterior and the problem with
the highest predicted learning gain wins. The reward is the mastery gain in the
problem's weakest topic.</p>
{recommendation_cards(state, now)}
</section>

<section>
<h2>Mastery across topics</h2>
<p>Weakest right now: {html.escape(weakest5)}. Strongest: {html.escape(strongest5)}.
New topics start at 50%: maximum uncertainty.</p>
{divs['mastery']}
</section>

<section>
<h2>Forgetting, visualized</h2>
<p>Mastery decays exponentially toward the 50% prior with a 30-day half-life.
Pick a topic from the dropdown to project its next 180 days. Stored counts are
never mutated, so raw evidence stays intact.</p>
{divs['forgetting']}
</section>

<section>
<h2>What the bandit learns from one log</h2>
<p>{html.escape(effect_note)}</p>
{divs['effect']}
</section>

<section>
<h2>The problem bank</h2>
{divs['difficulty']}
{divs['bank']}
</section>

<footer>
Generated {stamp} by <code>scripts/make_dashboard.py</code> from the real
grindbandit engine. Regenerate with <code>pip install plotly &amp;&amp;
python scripts/make_dashboard.py</code>. Plotly.js is bundled inside this
file, so the page works offline with no backend.
</footer>
</body>
</html>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
