# Changelog

## [Unreleased]

### Fixed
- Recommendation explanations now match the actual sampled draw (`src/grindbandit/bandit.py`). The old explanation derived its driver claim from the posterior mean while selection used a sampled theta, so it could credit a factor that did not drive the realized draw. The logged recommendation record now carries a sampled explanation: per-feature sampled contributions that sum with the intercept to the logged sampled score, the driver equal to the largest sampled contribution, and score uncertainty from 10,000 posterior resamples with the seed recorded. Picks made while the posterior score spread is still wide are labeled exploration, and uncertainty is reported as a model estimate, never as observed proficiency. The `next` command prints the logged explanation lines, and the demo-data helper passes its sampled theta to the explanation instead of recomputing from the mean.

## [v1.2.0] - 2026-10-30
### Docs
- Centered the README header: uniform dark badge labels and three action buttons in the dashboard palette (Open Dashboard, Explore Method, Run Locally).

### Added
- Recommendation logging (`src/grindbandit/bandit.py`, `state.py`, `cli.py`). Every `next` pick now appends a self-contained record to the state file: feature version (`context-v1`), the RNG seed used for the Thompson draw, the posterior and per-topic snapshot, and the full candidate set with context vectors and Monte Carlo selection probability estimates (10,000 posterior resamples on a separate RNG stream, add-one smoothed). `bandit.replay_selection` reconstructs the pick from the record alone, and tests pin that the replay matches the logged selection, before and after a JSON round trip.
- Attempt entries link back to the recommendation that selected the problem (`recommendation_id`) and label the reward as model-derived (`reward_kind: model_derived_mastery_change`); the `log` output uses the same wording.
- Delayed transfer quizzes. Logging an attempt schedules a quiz on the attempt's weakest topic with a different problem, due in 3 days. The new `quiz` command records the outcome into a separate `delayed_quizzes` store; outcomes never change mastery, the bandit posterior, or rewards, and `stats` lists pending quizzes. State files written before this change keep working.

## [v1.1.0] - 2026-10-18
### Fixed
- Forgetting no longer revives on update (`src/grindbandit/mastery.py`). `update` incremented raw counts and reset the decay clock, so after ten successes and 90 idle days (mastery 0.6923) a single failure raised modeled mastery to 0.8462 by resurrecting all the decayed successes. `update` now persists the decayed effective counts at the update time *before* adding the new observation; the raw attempt history stays intact in the append-only log. `replay` accepts the configured half-life so the weekly review reconstructs with the same decay as live mastery. Added regression tests: a failure after decay cannot raise the posterior mean, stored counts are decayed before the new observation lands, and incremental updates agree with chronological log replay across gaps and half-lives.
