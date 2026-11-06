"""Local persistence. Everything lives in one JSON file.

Besides topics, problems, the attempt log, and the bandit posterior, the
state holds two append-only records: `recommendations` (one full record
per pick, so any recommendation can be reconstructed and audited) and
`delayed_quizzes` (delayed transfer quiz schedules and their outcomes).
Delayed quiz outcomes are observational records: they live only in the
quiz store and are never fed into mastery, the bandit posterior, or the
model-derived reward on attempt entries.

Attempt/recommendation lineage is precise rather than reconstructed:
every attempt entry carries its own immutable `attempt_id`, a link to a
recommendation is either explicit (a user-supplied id, validated
against the chosen problem and rejected when stale or consumed) or
inferred (the recent, globally latest, unconsumed pick, labeled as
inferred on the entry), and each quiz names its source attempt by id.
Linkage fields are bookkeeping only and never feed the model.

Persistence guarantees (GB2):

- Saves are atomic: the new content is written to a temporary file in
  the same directory and moved over the old file with os.replace, so
  an interrupted serialization leaves the prior file readable.
- Saves are serialized: a save takes an exclusive lock on a
  `<path>.lock` file first. A save that cannot take the lock within
  one second fails with StateConflictError instead of silently
  overwriting another process's work, so two competing attempts are
  either both retained (sequential saves) or the second one gets an
  explicit conflict.
- Loads validate the schema and migrate deterministically: a state
  file written by an older version gains the missing top-level keys
  with their empty defaults, in memory only, and nothing else about
  the file changes. A file with the wrong shape is rejected with
  ValueError rather than loaded and found broken later.
- Bare filenames work: a path with no directory part saves and loads
  in the current directory.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

# Days between logging an attempt and its delayed transfer quiz coming due.
DELAYED_QUIZ_DELAY_DAYS = 3

# How recent the latest recommendation must be for `log` without an
# explicit id to infer that the attempt answers it. A day matches the
# daily practice cadence: a pick from an earlier session is a weak
# basis for claiming the link, so older recommendations are never
# auto-linked (an explicit --recommendation-id can still name the
# latest unconsumed pick, however old it is).
ATTEMPT_LINK_WINDOW_SECONDS = 86400.0

# How long a save waits for another process to release the state lock
# before failing with an explicit conflict instead of overwriting.
LOCK_TIMEOUT_SECONDS = 1.0

# Top-level state keys and the type each one must hold. A state file
# written before a key existed (for example, before recommendations
# or delayed quizzes were added) migrates by gaining that key with
# its empty default; the migration is the same on every load.
STATE_SCHEMA = {
    "topics": dict,
    "problems": dict,
    "log": list,
    "bandit": dict,
    "recommendations": list,
    "delayed_quizzes": list,
}

# Keys whose list entries must each be an object.
STATE_ENTRY_KEYS = ("log", "recommendations", "delayed_quizzes")


class StateConflictError(RuntimeError):
    """Raised when another process holds the state lock.

    The caller gets an explicit conflict instead of a silent
    overwrite: retry after the other process finishes, or reload and
    save again so both attempts are retained.
    """


def _blank_state() -> dict:
    return {
        "topics": {},
        "problems": {},
        "log": [],
        "bandit": {},
        "recommendations": [],
        "delayed_quizzes": [],
    }


def migrate_state(state: dict) -> dict:
    """Return a copy of `state` with every schema key present.

    Missing keys gain their empty defaults; keys already present keep
    their values untouched, and unknown keys are kept as they are.
    The result is the same on every call for the same input, and the
    file on disk is not touched: migration happens in memory on load.
    """
    migrated = copy.deepcopy(state)
    for key, expected in STATE_SCHEMA.items():
        if key not in migrated:
            migrated[key] = copy.deepcopy(_blank_state()[key])
    return migrated


def validate_state(state: dict) -> dict:
    """Raise ValueError when `state` does not match STATE_SCHEMA.

    Returns the state unchanged so callers can chain the check. A
    non-dict state, a key holding the wrong type, or a list entry
    that is not an object is rejected by name, before anything is
    written or relied on.
    """
    if not isinstance(state, dict):
        raise ValueError(
            f"state must be an object, got {type(state).__name__}"
        )
    for key, expected in STATE_SCHEMA.items():
        value = state.get(key)
        if not isinstance(value, expected):
            raise ValueError(
                f"state key {key!r} must be a {expected.__name__}, "
                f"got {type(value).__name__ if value is not None else 'missing'}"
            )
    for key in STATE_ENTRY_KEYS:
        for entry in state[key]:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"state key {key!r} must hold objects, "
                    f"got {type(entry).__name__}"
                )
    return state


def _lock_path(path: str) -> str:
    return path + ".lock"


def _acquire_lock(path: str):
    """Take an exclusive lock on the `<path>.lock` file.

    Returns the open lock file; closing it releases the lock. Waits up
    to LOCK_TIMEOUT_SECONDS, then raises StateConflictError so a
    contended save fails loudly instead of overwriting.
    """
    lock_file = open(_lock_path(path), "a+")
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except BlockingIOError:
            if time.monotonic() >= deadline:
                lock_file.close()
                raise StateConflictError(
                    f"state file {path} is locked by another process; "
                    "reload and save again instead of overwriting"
                )
            time.sleep(0.05)


def _fsync_dir(directory: str) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def default_path() -> str:
    return os.environ.get(
        "GRIND_BANDIT_STATE", str(Path.home() / ".grind-bandit" / "state.json")
    )


def load(path: str | None = None) -> dict:
    path = path or default_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"state file {path} is not valid JSON: {exc}"
            ) from exc
        return validate_state(migrate_state(raw))
    return _blank_state()


def save(state: dict, path: str | None = None) -> str:
    path = path or default_path()
    state = validate_state(migrate_state(state))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    lock_file = _acquire_lock(path)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, prefix=".state-", suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            json.dump(state, tmp, indent=1)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
        _fsync_dir(directory)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
        lock_file.close()
    return path


def record_attempt(
    state: dict,
    problem: dict,
    solved: bool,
    minutes: int,
    reward: float,
    now: float,
    recommendation_id: str | None = None,
    linkage_method: str | None = None,
    recommendation_status: dict | None = None,
    attempt_id: str | None = None,
) -> dict:
    """Append an attempt entry and return it.

    Every entry carries an immutable attempt_id, unique per attempt,
    so downstream records (delayed quizzes) can name the exact attempt
    they came from and two attempts on the same problem never alias one
    source. A fresh id is generated unless the caller supplies one
    (the demo-data builder does, so its seeded history stays
    reproducible). When the attempt answers a recommendation, the
    caller passes the validated recommendation_id together with how the
    link was established: linkage_method is "explicit" when the user
    named the id, "inferred" when it was auto-linked from the latest
    unconsumed pick. recommendation_status records, where known,
    whether the recommendation was presented ("exposed") and whether
    this attempt is the pick being taken up ("accepted"). Linkage
    fields are bookkeeping only: they are written on the entry and
    nowhere else, and never feed mastery, the bandit posterior, or the
    reward, which is computed by the caller before this runs.
    """
    entry = {
        "attempt_id": attempt_id or f"att-{uuid.uuid4().hex[:12]}",
        "ts": now,
        "problem_id": problem["id"],
        "title": problem["title"],
        "difficulty": problem["difficulty"],
        "topics": problem["topics"],
        "solved": solved,
        "minutes": minutes,
        "reward": reward,
        # The reward is computed by the mastery model itself (change in
        # modeled mastery of the weakest topic). It is labeled as such so
        # it is never mistaken for a measured learning outcome.
        "reward_kind": "model_derived_mastery_change",
    }
    if recommendation_id is not None:
        entry["recommendation_id"] = recommendation_id
        entry["linkage_method"] = linkage_method or "explicit"
        if recommendation_status is not None:
            entry["recommendation_status"] = recommendation_status
    state.setdefault("log", []).append(entry)
    p = state.setdefault("problems", {}).setdefault(
        str(problem["id"]),
        {"attempts": 0, "solved": 0, "last_ts": None, "ever_solved": False},
    )
    p["attempts"] += 1
    p["last_ts"] = now
    if solved:
        p["solved"] += 1
        p["ever_solved"] = True
    return entry


def append_recommendation(state: dict, record: dict) -> None:
    state.setdefault("recommendations", []).append(record)


def latest_recommendation_id(state: dict, problem_id: int) -> str | None:
    """Id of the most recent recommendation that selected this problem.

    Kept for reading old records. New attempt logging must not use it
    to claim a link: it matches on problem id alone, so it cannot tell
    a confirmed answer to the latest pick from a problem the user chose
    themselves that happens to have been recommended at some point. Use
    validate_recommendation_link / infer_recommendation_link instead.
    """
    for rec in reversed(state.get("recommendations", [])):
        if rec.get("selected_problem_id") == problem_id:
            return rec.get("id")
    return None


def find_recommendation(state: dict, recommendation_id: str) -> dict | None:
    for rec in state.get("recommendations", []):
        if rec.get("id") == recommendation_id:
            return rec
    return None


def recommendation_consumed(state: dict, recommendation_id: str) -> bool:
    """True when a logged attempt already links to this recommendation.

    Consumption is derived from the attempt log itself, so it needs no
    extra bookkeeping and stays true to history: a pick is taken up at
    most once, and a second attempt cannot silently re-link to it.
    """
    return any(
        e.get("recommendation_id") == recommendation_id
        for e in state.get("log", [])
    )


def validate_recommendation_link(
    state: dict, recommendation_id: str, problem_id: int
) -> dict:
    """Return the recommendation an explicit id may link to, or raise.

    The id must exist, must have selected the problem being logged,
    and must still be the latest unconsumed recommendation: a link to
    an older pick is stale once a newer recommendation exists, because
    the newer pick is the decision the user was actually presented
    with most recently. Raises ValueError on any mismatch; callers
    validate before mutating anything, so a rejection logs nothing.
    """
    rec = find_recommendation(state, recommendation_id)
    if rec is None:
        raise ValueError(f"unknown recommendation id: {recommendation_id}")
    if rec.get("selected_problem_id") != problem_id:
        raise ValueError(
            f"recommendation {recommendation_id} selected problem "
            f"{rec.get('selected_problem_id')}, not problem {problem_id}"
        )
    recs = state.get("recommendations", [])
    if not recs or recs[-1].get("id") != recommendation_id:
        raise ValueError(
            f"recommendation {recommendation_id} is stale: a newer "
            "recommendation has been made since"
        )
    if recommendation_consumed(state, recommendation_id):
        raise ValueError(
            f"recommendation {recommendation_id} was already taken up "
            "by an earlier attempt"
        )
    return rec


def infer_recommendation_link(
    state: dict,
    problem_id: int,
    now: float,
    window: float = ATTEMPT_LINK_WINDOW_SECONDS,
) -> dict | None:
    """The recommendation an attempt may be inferred to answer, or None.

    Inference is deliberately narrow: only the globally latest
    recommendation, only when it selected this problem, only while it
    is unconsumed, and only within `window` of being made. Anything
    older, superseded, consumed, or for another problem means the
    attempt stands as manually chosen and unlinked. Callers record an
    inferred link as inferred, never as confirmed.
    """
    recs = state.get("recommendations", [])
    if not recs:
        return None
    rec = recs[-1]
    if rec.get("selected_problem_id") != problem_id:
        return None
    if recommendation_consumed(state, rec.get("id")):
        return None
    ts = rec.get("ts")
    if ts is None or not 0.0 <= now - ts <= window:
        return None
    return rec


def _assigned_quiz_problem_ids(state: dict) -> list:
    """Problem ids assigned to earlier quizzes, in assignment order.

    Together with the quiz records themselves this is the exposure
    history: how many times a problem has been handed out as a quiz is
    recoverable from the store, and each new record also carries its
    own count (see schedule_delayed_quiz).
    """
    return [q.get("quiz_problem_id") for q in state.get("delayed_quizzes", [])]


def _quiz_pool(bank: list[dict], attempted: dict, topic: str) -> list[dict]:
    return [p for p in bank if topic in p["topics"] and p["id"] != attempted["id"]]


def _is_novel(problem: dict, attempted_ids: set, assigned_ids: set) -> bool:
    """A problem is novel as a quiz item if it was never attempted and
    never assigned to an earlier quiz."""
    return str(problem["id"]) not in attempted_ids and problem["id"] not in assigned_ids


def choose_quiz_problem(
    bank: list[dict], attempted: dict, topic: str, state: dict
) -> dict | None:
    """Pick a different problem on the same topic for a delayed quiz.

    Deterministic priority, lowest id inside each tier:

    1. novel problems: never attempted and never assigned as a quiz;
    2. assigned before but never attempted;
    3. attempted before but never assigned;
    4. attempted and assigned before.

    A problem is therefore never assigned twice while a novel
    alternative exists, and previously attempted problems are the last
    resort. When only reused problems remain the pick is still made;
    schedule_delayed_quiz states that exhaustion on the record.
    """
    pool = _quiz_pool(bank, attempted, topic)
    if not pool:
        return None
    attempted_ids = set(state.get("problems", {}).keys())
    assigned_ids = {i for i in _assigned_quiz_problem_ids(state) if i is not None}

    def tier(p: dict) -> tuple:
        return (str(p["id"]) in attempted_ids, p["id"] in assigned_ids)

    return min(pool, key=lambda p: (tier(p), p["id"]))


def schedule_delayed_quiz(
    state: dict,
    bank: list[dict],
    attempted: dict,
    topic: str,
    now: float,
    recommendation_id: str | None = None,
    source_attempt_id: str | None = None,
) -> dict | None:
    """Schedule a delayed transfer quiz for an attempt. Returns it or None.

    The quiz targets the attempt's weakest topic with a different
    problem and comes due DELAYED_QUIZ_DELAY_DAYS later. It is only a
    schedule: the outcome, when recorded, lands in this same store and
    nowhere else. source_attempt_id is the immutable id of the exact
    attempt that produced this quiz (see record_attempt), recorded
    alongside the source problem and recommendation so the lineage
    names one attempt, never just a problem.
    """
    quiz_problem = choose_quiz_problem(bank, attempted, topic, state)
    if quiz_problem is None:
        return None
    assigned_ids = _assigned_quiz_problem_ids(state)
    attempted_ids = set(state.get("problems", {}).keys())
    novel_pool = [
        p for p in _quiz_pool(bank, attempted, topic)
        if _is_novel(p, attempted_ids, set(assigned_ids))
    ]
    quiz = {
        "id": f"quiz-{uuid.uuid4().hex[:12]}",
        "kind": "delayed_transfer_quiz",
        "scheduled_ts": now,
        "due_ts": now + DELAYED_QUIZ_DELAY_DAYS * 86400.0,
        "delay_days": DELAYED_QUIZ_DELAY_DAYS,
        "topic": topic,
        "source_problem_id": attempted["id"],
        "source_attempt_id": source_attempt_id,
        "source_recommendation_id": recommendation_id,
        "quiz_problem_id": quiz_problem["id"],
        "quiz_title": quiz_problem["title"],
        # How this problem was chosen: whether it is novel (never
        # attempted, never assigned), how many times it was assigned
        # to earlier quizzes, and whether the novel pool was already
        # exhausted, in which case this assignment is an explicit
        # reuse rather than a silent repeat.
        "assignment": {
            "novel": _is_novel(quiz_problem, attempted_ids, set(assigned_ids)),
            "prior_assignments": assigned_ids.count(quiz_problem["id"]),
            "pool_exhausted": not novel_pool,
        },
        "status": "pending",
        "outcome": None,
    }
    state.setdefault("delayed_quizzes", []).append(quiz)
    return quiz


def find_delayed_quiz(state: dict, quiz_id: str) -> dict | None:
    for q in state.get("delayed_quizzes", []):
        if q.get("id") == quiz_id:
            return q
    return None


def record_delayed_outcome(
    state: dict, quiz_id: str, solved: bool, now: float, allow_early: bool = False
) -> dict | None:
    """Store a delayed quiz outcome in the quiz store only.

    Deliberately touches nothing else: the outcome is an observational
    record, kept separate from attempts, mastery, and the bandit
    posterior, and it never feeds the model-derived reward.

    The quiz measures retention after its delay, so an outcome dated
    before the due date is rejected with a ValueError unless
    allow_early is passed explicitly. A record with no due date (none
    are written that way anymore) is treated as due. The stored
    outcome carries the realized delay in seconds (scheduled to
    recorded) and an early / on-time / late status: early is before
    the due date, on-time is the due day itself, late is after it.
    """
    quiz = find_delayed_quiz(state, quiz_id)
    if quiz is None or quiz.get("status") != "pending":
        return None
    due_ts = quiz.get("due_ts")
    if due_ts is not None and now < due_ts and not allow_early:
        raise ValueError(
            f"delayed quiz {quiz_id} has not come due yet; an outcome "
            "recorded before the due date does not measure retention "
            "after the delay (pass allow_early=True to record it anyway)"
        )
    scheduled_ts = quiz.get("scheduled_ts")
    if due_ts is not None and now < due_ts:
        timing = "early"
    elif due_ts is not None and now > due_ts + 86400.0:
        timing = "late"
    else:
        timing = "on_time"
    quiz["status"] = "completed"
    quiz["outcome"] = {
        "solved": bool(solved),
        "recorded_ts": now,
        "delay_seconds": None if scheduled_ts is None else now - scheduled_ts,
        "timing": timing,
    }
    return quiz


def pending_delayed_quizzes(state: dict) -> list[dict]:
    return [q for q in state.get("delayed_quizzes", []) if q.get("status") == "pending"]
