# Changelog

## [v1.1.0] - 2026-10-18
### Fixed
- Forgetting no longer revives on update (`src/grindbandit/mastery.py`). `update` incremented raw counts and reset the decay clock, so after ten successes and 90 idle days (mastery 0.6923) a single failure raised modeled mastery to 0.8462 by resurrecting all the decayed successes. `update` now persists the decayed effective counts at the update time *before* adding the new observation; the raw attempt history stays intact in the append-only log. `replay` accepts the configured half-life so the weekly review reconstructs with the same decay as live mastery. Added regression tests: a failure after decay cannot raise the posterior mean, stored counts are decayed before the new observation lands, and incremental updates agree with chronological log replay across gaps and half-lives.
