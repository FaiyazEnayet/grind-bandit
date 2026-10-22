"""Command line interface: next, log, quiz, stats, review, export, reset."""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
import time
from datetime import datetime, timezone

from . import bandit, mastery, problems, state as st

BLOCKS = ["\u2591", "\u2592", "\u2593", "\u2588"]  # light to full: mastery shading


def _now() -> float:
    return time.time()


def _topic_last(state: dict, topic: str):
    return state.get("topics", {}).get(topic, {}).get("last_ts")


def select_candidates(bank: list[dict], state: dict, now: float):
    """Unsolved problems not attempted in the last day.

    Returns (candidates, review_mode). When everything is solved we fall
    back to the full bank and say so honestly: that is review mode.
    """
    probs = state.get("problems", {})
    cands = []
    for p in bank:
        rec = probs.get(str(p["id"]), {})
        if rec.get("ever_solved"):
            continue
        last = rec.get("last_ts")
        if last is not None and now - last < 86400.0:
            continue
        cands.append(p)
    if not cands:
        return bank, True
    return cands, False


def cmd_next(args) -> None:
    state = st.load()
    bank = problems.load()
    now = _now()
    cands, review_mode = select_candidates(bank, state, now)
    # Record the seed that drives the draw. When the user does not pass
    # one, generate it here so the logged recommendation still carries
    # the exact seed and can be reconstructed later.
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    record = bandit.build_recommendation(
        state, cands, now, seed=seed, review_mode=review_mode
    )
    st.append_recommendation(state, record)
    st.save(state)
    p = problems.by_id(bank, record["selected_problem_id"])
    print(f"#{p['id']} {p['title']} ({p['difficulty']})")
    print(problems.url(p))
    print(f"Topics: {', '.join(p['topics'])}")
    if review_mode:
        print("Note: you solved every problem in the bank. Review mode: pick anything.")
    print("Why this one:")
    for line in bandit.explain(p, state, now):
        print(f"  * {line}")


def cmd_log(args) -> None:
    state = st.load()
    bank = problems.load()
    p = problems.by_id(bank, args.problem_id)
    if p is None:
        print(f"unknown problem id: {args.problem_id}", file=sys.stderr)
        raise SystemExit(1)
    solved = args.result == "solved"
    now = _now()
    x = bandit.context(p, state, now)
    tm = {t: mastery.mean(state, t, now) for t in p["topics"]}
    weak = bandit.weakest_topic(p, tm)
    before = tm[weak]
    mastery.update(state, p["topics"], solved, now)
    after = mastery.mean(state, weak, now)
    reward = max(-1.0, min(1.0, after - before))
    bandit.observe(state, x, reward)
    rec_id = st.latest_recommendation_id(state, p["id"])
    st.record_attempt(state, p, solved, args.minutes, reward, now,
                      recommendation_id=rec_id)
    quiz = st.schedule_delayed_quiz(state, bank, p, weak, now,
                                    recommendation_id=rec_id)
    st.save(state)
    print(f"logged #{p['id']} {p['title']}: {args.result} ({args.minutes} min)")
    print(f"model-derived reward: {reward:+.3f} (change in modeled mastery "
          f"of {weak}; computed by the model, not a measured learning gain)")
    if quiz is not None:
        due = datetime.fromtimestamp(quiz["due_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"delayed transfer quiz scheduled for {due}: "
              f"#{quiz['quiz_problem_id']} {quiz['quiz_title']} ({weak}). "
              f"Its outcome is recorded separately and never feeds the model.")


def cmd_quiz(args) -> None:
    state = st.load()
    quiz = st.find_delayed_quiz(state, args.quiz_id)
    if quiz is None:
        print(f"unknown quiz id: {args.quiz_id}", file=sys.stderr)
        raise SystemExit(1)
    if quiz["status"] != "pending":
        print(f"quiz {args.quiz_id} already has a recorded outcome", file=sys.stderr)
        raise SystemExit(1)
    st.record_delayed_outcome(state, args.quiz_id, args.result == "solved", _now())
    st.save(state)
    print(f"recorded delayed quiz {args.quiz_id}: {args.result}")
    print("stored as an observational outcome only: it does not change "
          "mastery, the bandit posterior, or any reward")


def _heatmap(state: dict, bank: list[dict], now: float) -> list[str]:
    topics = problems.all_topics(bank)
    cols = 4
    width = 22
    lines = []
    for i in range(0, len(topics), cols):
        row = topics[i : i + cols]
        lines.append("+" + "+".join(["-" * width] * len(row)) + "+")
        lines.append(
            "|" + "|".join(f" {t:<{width - 2}} " for t in row) + "|"
        )
        cells = []
        for t in row:
            m = mastery.mean(state, t, now)
            shade = BLOCKS[min(3, int(m * 4))]
            cells.append(f" {shade * 8} {m:>4.0%}      ")
        lines.append("|" + "|".join(cells) + "|")
    lines.append("+" + "+".join(["-" * width] * cols) + "+")
    return lines


def cmd_stats(args) -> None:
    state = st.load()
    bank = problems.load()
    now = _now()
    print("Mastery heatmap (darker blocks mean stronger mastery):")
    for line in _heatmap(state, bank, now):
        print(line)
    print()
    print(f"{'topic':<18}{'mastery':<9}{'tried':<8}{'solved':<9}{'last practiced'}")
    tm = mastery.mastery_map(state, now)
    topics = state.get("topics", {})
    for t in sorted(topics, key=lambda t: mastery.mean(state, t, now)):
        m = mastery.mean(state, t, now)
        info = topics[t]
        last = info.get("last_ts")
        if last is None:
            ago = "never"
        else:
            d = (now - last) / 86400.0
            ago = "today" if d < 1 else f"{int(d)}d ago"
        rate = info["solved"] / info["attempts"] if info["attempts"] else 0.0
        bar = "\u2588" * int(m * 10) + "\u2591" * (10 - int(m * 10))
        print(f"{t:<18}{bar} {m:<4.0%}{info['attempts']:<8}{rate:<9.0%}{ago}")
    pending = st.pending_delayed_quizzes(state)
    if pending:
        print()
        print("Delayed transfer quizzes pending (outcomes are observational "
              "and kept separate from the model):")
        for q in sorted(pending, key=lambda q: q["due_ts"]):
            due = datetime.fromtimestamp(q["due_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            tag = "due now" if q["due_ts"] <= now else f"due {due}"
            print(f"  {q['id']}: #{q['quiz_problem_id']} {q['quiz_title']} "
                  f"({q['topic']}, {tag})")
            print(f"    record with: grind-bandit quiz {q['id']} solved|failed")


def cmd_review(args) -> None:
    state = st.load()
    log = state.get("log", [])
    if not log:
        print("nothing logged yet. Log some attempts first.")
        return
    now = _now()
    weeks = args.weeks
    hl = mastery.half_life_days(state)
    print(f"Weekly review: last {weeks} week(s)")
    for w in range(weeks):
        end = now - w * 7 * 86400.0
        start = end - 7 * 86400.0
        entries = [e for e in log if start < e["ts"] <= end]
        if not entries:
            continue
        label = datetime.fromtimestamp(end, tz=timezone.utc).strftime("%Y-%m-%d")
        n = len(entries)
        solved = sum(1 for e in entries if e["solved"])
        minutes = sum(e["minutes"] for e in entries)
        print(f"\nweek ending {label}: {n} attempts, {solved} solved ({solved / n:.0%}), {minutes} min")
        before = mastery.replay(log, start, hl)
        after = mastery.replay(log, end, hl)
        deltas = {
            t: after.get(t, 0.5) - before.get(t, 0.5)
            for t in set(after) | set(before)
        }
        movers = sorted(deltas.items(), key=lambda kv: -kv[1])
        if movers:
            t, d = movers[0]
            print(f"  most improved: {t} ({before.get(t, 0.5):.0%} -> {after.get(t, 0.5):.0%})")
        weak = min(after, key=lambda t: after[t])
        print(f"  weakest now: {weak} ({after[weak]:.0%})")


def cmd_export(args) -> None:
    state = st.load()
    fmt = args.format
    out = args.out
    if fmt == "json":
        with open(out, "w") as f:
            json.dump(state, f, indent=1)
    else:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "date", "problem_id", "title", "difficulty",
                        "topics", "solved", "minutes", "reward"])
            for e in state.get("log", []):
                w.writerow([
                    e["ts"],
                    datetime.fromtimestamp(e["ts"], tz=timezone.utc).strftime("%Y-%m-%d"),
                    e["problem_id"], e["title"], e["difficulty"],
                    ";".join(e["topics"]), e["solved"], e["minutes"],
                    f"{e['reward']:.4f}",
                ])
    print(f"exported {len(state.get('log', []))} attempts to {out}")


def cmd_reset(args) -> None:
    if not args.yes:
        ans = input("wipe all progress? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            print("aborted")
            return
    st.save({"topics": {}, "problems": {}, "log": [], "bandit": {},
             "recommendations": [], "delayed_quizzes": []})
    print("progress wiped")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="grind-bandit",
                                 description="Your LeetCode grind as a contextual bandit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("next", help="recommend the next problem")
    n.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    n.set_defaults(func=cmd_next)

    lg = sub.add_parser("log", help="log an attempt")
    lg.add_argument("problem_id", type=int)
    lg.add_argument("result", choices=["solved", "failed"])
    lg.add_argument("--minutes", type=int, default=30)
    lg.set_defaults(func=cmd_log)

    qz = sub.add_parser("quiz", help="record the outcome of a delayed transfer quiz")
    qz.add_argument("quiz_id")
    qz.add_argument("result", choices=["solved", "failed"])
    qz.set_defaults(func=cmd_quiz)

    st_ = sub.add_parser("stats", help="mastery dashboard")
    st_.set_defaults(func=cmd_stats)

    rv = sub.add_parser("review", help="weekly mastery review")
    rv.add_argument("--weeks", type=int, default=4)
    rv.set_defaults(func=cmd_review)

    ex = sub.add_parser("export", help="dump history to CSV or JSON")
    ex.add_argument("--format", choices=["csv", "json"], default="csv")
    ex.add_argument("--out", required=True)
    ex.set_defaults(func=cmd_export)

    rs = sub.add_parser("reset", help="wipe all progress")
    rs.add_argument("--yes", action="store_true")
    rs.set_defaults(func=cmd_reset)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
