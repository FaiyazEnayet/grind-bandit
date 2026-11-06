# State schema and upgrade register

The state file (`~/.grind-bandit/state.json` by default, or
`$GRIND_BANDIT_STATE`) is one JSON object. This page records every
schema version, what changed, and how an older file migrates. It is
the register of record: when the on-disk shape changes, the new
version is recorded here first, with the migration that bridges it.

## Current shape (version 1)

| Key | Type | Contents |
| :--- | :--- | :--- |
| `schema_version` | int | Schema version this file was written under (1). |
| `topics` | object | Per-topic Beta-Bernoulli counts (`alpha`, `beta`, `attempts`, `solved`, `last_ts`). |
| `problems` | object | Per-problem attempt counts and `ever_solved`. |
| `log` | list | Attempt entries, oldest first. |
| `bandit` | object | Thompson sampling posterior (`mean`, `prec`). |
| `recommendations` | list | One full record per `next` pick. |
| `delayed_quizzes` | list | Delayed transfer quiz schedules and outcomes. |

Attempt entries carry an immutable `attempt_id` and, where linked,
a `recommendation_id` with `linkage_method` (`explicit` or
`inferred`). Those fields exist only on entries logged after they
were introduced; old entries are never given them retroactively.

## Version history

### Version 0: before versions existed

Any file with no `schema_version` key. Written by releases up to
and including v1.1.0. It has four top-level keys only (`topics`,
`problems`, `log`, `bandit`): no `recommendations` or
`delayed_quizzes` keys, and its log entries have no `attempt_id`,
no `reward_kind`, and no recommendation linkage. The values are the
same shapes as version 1; only the newer keys and fields are
absent.

### Version 1: current

Adds `schema_version: 1`, the `recommendations` and
`delayed_quizzes` top-level keys, and per-attempt lineage
(`attempt_id`, `recommendation_id`, `linkage_method`,
`recommendation_status`). Built up across the v1.2.0 work:
recommendation logging, delayed transfer quizzes, and immutable
attempt ids. Nothing else about the stored values changed.

## Migration rules

Migration is a pure function of the file contents (`state.migrate_state`):
the same input always produces the same output.

- **On load, in memory only.** `state.load` migrates the loaded
  dict and validates it, but never writes the file. A version 0
  file on disk stays a version 0 file until it is migrated on
  purpose. Missing keys gain their empty defaults; present values
  are kept untouched, including unknown keys.
- **Persisted, with backup.** `grind-bandit migrate [--path PATH]`
  (or `state.migrate_file`) writes the migrated state back. Before
  writing, it copies the exact original bytes to
  `<path>.v<from>.bak` (for example `state.json.v0.bak`). The
  backup is written once: if one already exists it is left alone,
  so the earliest pre-migration copy is the one kept. The migrated
  file is written atomically, under the same lock as any save.
- **Already current is a no-op.** Migrating a version 1 file
  writes nothing and makes no backup.
- **A newer version is refused.** A file whose `schema_version`
  is higher than the code supports is rejected with a `ValueError`
  on load and on migrate, and left untouched. It is never silently
  downgraded.
- **A wrongly shaped file is rejected.** A non-object file, a key
  holding the wrong type, or a list entry that is not an object
  raises `ValueError` by name, before anything is written or
  relied on.

### What migration never invents

Migration adds structure, never evidence. An old log entry gains
no `attempt_id`, no recommendation link, and no `linkage_method`:
entries logged before those fields existed stay unlinked, and
their quizzes (if any were scheduled later) gain no source attempt
id. The `recommendations` and `delayed_quizzes` keys arrive as
empty lists, never with fabricated records. Topic counts, problem
records, log entries, and the bandit posterior pass through
unchanged. The demo-data builder (`scripts/common.py`,
`build_demo_state`) returns a version 0 shaped state, and it
migrates under exactly these rules.

## Old example state

`tests/fixtures/state-v0-example.json` is a version 0 state file:
three scripted attempts (Two Sum solved, Add Two Numbers failed,
Find the Index of the First Occurrence in a String solved) on real
bank problems, with topic counts and a bandit posterior computed by
the engine for exactly those attempts. It is a scripted example,
not a real user's history. It exists so the migration path is
tested against a real old-format file, not a dict built inline in
a test. `tests/test_stage3_grind_bandit_7.py` migrates it twice and
checks the results are identical, that the backup holds the exact
original bytes, and that no history is invented.
