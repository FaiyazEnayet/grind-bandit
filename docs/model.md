# The model, worked out

Grind-Bandit has two models that feed each other. Mastery tracks
what you know per topic. The bandit picks the next problem. The
reward that trains the bandit is a change in mastery, so the two
models are linked by construction. Every number in this page is a
worked example computed by the engine, and every one of them is
reproduced by a test in `tests/test_stage3_grind_bandit_7.py`. If a
number here disagrees with a test, the test is right and this page
is stale.

## Mastery: Beta-Bernoulli with forgetting

Each topic holds wins (`alpha`) and losses (`beta`). Mastery is the
posterior mean:

    mastery = alpha / (alpha + beta)

A new topic starts at alpha = beta = 1, so mastery is 0.5: maximum
uncertainty. A solved attempt adds one win to every topic on the
problem; a failed attempt adds one loss.

### Forgetting

Stored counts decay exponentially toward the (1, 1) prior based on
time since the topic was last attempted. With a 30-day half-life
(`mastery.HALF_LIFE_DAYS`, configurable per state via
`config.half_life_days`) and `w = 0.5 ** (days / 30)`:

    alpha_effective = 1 + (alpha - 1) * w
    beta_effective  = 1 + (beta - 1) * w

Decay is persisted on every update *before* the new observation is
added, so a new attempt can never revive long-forgotten evidence.
The raw attempt history stays intact in the append-only log.

### Worked example: ten solves, ninety idle days, one failure

Start from a fresh topic (`alpha = 1, beta = 1`, mastery 0.5) and
log ten solves at the same timestamp:

- After the ten solves: `alpha = 11, beta = 1`, mastery `11 / 12 = 0.9167`.
- Ninety idle days later, `w = 0.5 ** 3 = 0.125`, so the effective
  counts are `alpha = 1 + 10 * 0.125 = 2.25` and `beta = 1`, and
  mastery is `2.25 / 3.25 = 0.6923`.
- Log a failure at that point. Decay is persisted first (2.25, 1.0),
  then the loss lands: `alpha = 2.25, beta = 2.0`, mastery
  `2.25 / 4.25 = 0.5294`. The failure lowers mastery, as it should.

Before this was fixed, the same failure raised modeled mastery to
0.8462 by resurrecting the decayed successes. That is the GB1 fix,
recorded in the changelog under v1.1.0.

## Reward: change in weakest-topic mastery

The reward for an attempt is the change in mastery of the
problem's weakest topic, clipped to [-1, 1]:

    reward = clip(mastery_after - mastery_before, -1, 1)

Solving grows mastery (positive reward); failing shrinks it
(negative reward). The bandit learns which contexts produce
learning, not just which problems get solved.

### Worked example

On a fresh topic (mastery 0.5):

- Solve: mastery becomes `2 / 3 = 0.6667`, reward `+1/6 = +0.1667`.
- Fail: mastery becomes `1 / 3 = 0.3333`, reward `-1/6 = -0.1667`.

The reward is model-derived: it is computed by the same mastery
model the selector uses, so by itself it is not independent
evidence that a recommendation caused learning. That is why the
reward is labeled `model_derived_mastery_change` on every attempt
entry, and why delayed transfer quiz outcomes are stored separately
and never feed it.

## Context: what the bandit sees

Each candidate problem is a five-feature vector, all in [0, 1]:

| Feature | Meaning |
| :--- | :--- |
| `gap_min` | 1 minus mastery of the problem's weakest topic. |
| `gap_mean` | 1 minus mean mastery across the problem's topics. |
| `difficulty_match` | How well the problem's difficulty fits the level the weakest-topic mastery implies: below 0.4 implies Easy (0.0), below 0.7 Medium (0.5), otherwise Hard (1.0); the score is `1 - abs(problem_level - implied_level)`. |
| `recency` | Days since the least recently practiced topic on the problem, capped at 30, divided by 30. A never-practiced topic counts as the full 30 days. |
| `bias` | Constant 1, so the model learns a global offset. |

### Worked example

Problem 28, "Find the Index of the First Occurrence in a String"
(Easy, topic `string`), against a fresh state at the same
timestamp: mastery of `string` is 0.5, never practiced.

- `gap_min = 1 - 0.5 = 0.5`
- `gap_mean = 1 - 0.5 = 0.5`
- `difficulty_match`: mastery 0.5 implies Medium (0.5); Easy is
  0.0, so `1 - |0.0 - 0.5| = 0.5`
- `recency = 30 / 30 = 1.0` (never practiced)
- `bias = 1.0`

Context: `[0.5, 0.5, 0.5, 1.0, 1.0]`.

Under the older `context-v1` layout, recency measured days since
the *most* recently practiced topic and ignored never-practiced
topics entirely. Records logged under v1 still replay through the
preserved v1 path; current records are `context-v2`.

## The bandit: Thompson sampling over a linear model

Expected learning gain is linear in the context: `reward = theta . x`.
Theta has a Gaussian posterior that starts at zero (no assumptions)
and updates with every logged attempt. To pick, the bandit samples
one theta from the posterior and takes the argmax. Early on the
posterior is wide, so sampling explores; as evidence accumulates it
concentrates, so sampling exploits.

Posterior update for one observation `(x, r)`, with prior precision
`lambda = 1.0` and reward noise variance `sigma2 = 0.1`:

    prec_new = prec + x x^T / sigma2
    mean_new = prec_new^-1 (prec mean + x r / sigma2)

### Worked example: one observation

Prior mean `[0, 0, 0, 0, 0]`, prior precision the identity. Observe
`x = [0.5, 0.5, 0.5, 1.0, 1.0]` with reward `r = +1/6`:

- Posterior mean: `[0.02924, 0.02924, 0.02924, 0.05848, 0.05848]`
  (full precision in the test: 0.0292397661 and 0.0584795322).
- Posterior precision:

```
[[ 3.5, 2.5, 2.5,  5.0,  5.0],
 [ 2.5, 3.5, 2.5,  5.0,  5.0],
 [ 2.5, 2.5, 3.5,  5.0,  5.0],
 [ 5.0, 5.0, 5.0, 11.0, 10.0],
 [ 5.0, 5.0, 5.0, 10.0, 11.0]]
```

Observe the same context again with the opposite reward (`-1/6`)
and the posterior mean returns to exactly `[0, 0, 0, 0, 0]`: equal
and opposite evidence cancels, while the precision keeps growing.

### Worked example: a reproducible draw

Sampling theta from the fresh prior (mean zero, precision identity)
with seed 7 gives:

    theta = [0.001230, 0.298746, -0.274138, -0.890592, -0.454671]

The same seed on the same posterior always gives the same draw,
which is what makes a logged recommendation replayable: the record
carries the seed, the posterior snapshot, and every candidate's
context vector, so the pick can be reconstructed from the record
alone. Selection probabilities are Monte Carlo estimates from the
posterior (10,000 resamples on a separate RNG stream, add-one
smoothed): they say how likely the model was to pick each problem,
nothing more.

## Eligibility and spaced review

A problem is eligible when it is new practice (never solved, and
not attempted within the one-day attempt cooldown) or a due review
(solved before, and the review interval has elapsed since the last
attempt). The review interval defaults to 7 days
(`bandit.REVIEW_INTERVAL_DAYS`, configurable per state via
`config.review_interval_days`).

### Worked example

A problem solved 3 days ago: `review_cooldown`, not eligible ("solved
3.0 days ago; review comes due at 7 days since the last attempt").
The same problem solved 8 days ago: `due_review`, eligible ("solved
8.0 days ago; the 7-day review interval has elapsed").

## Delayed transfer quizzes

Logging an attempt schedules a quiz on the attempt's weakest topic
with a different problem, due 3 days later. The outcome is stored
in its own part of the state file and never feeds mastery, the
bandit posterior, or any reward. It is an observational record,
useful for later comparison, not causal evidence that a
recommendation improved learning.

## Honest limits

- The reward is model-derived, not a measured learning outcome.
- Selection probabilities and explanation uncertainty are model
  estimates, not measured frequencies or observed proficiency.
- The 7-day review interval and the 30-day forgetting half-life
  are modeling choices, not fitted optima. The interval is a
  starting heuristic for a daily practice habit; no retention
  measurement has been fitted to it.
- Cold-start recommendations depend heavily on the priors.
- Easy-only histories can bias estimated learning gain.
