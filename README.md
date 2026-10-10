# grind-bandit

Your LeetCode grind as a contextual bandit.

This is a tool to ease the finding of your next leetcode problem. It explores weak topics and exploits what is almost mastered. That tradeoff is handled by Bayes.

## The math

Each topic gets a Beta Bernoulli skill model. A solve adds a win, a fail adds a loss. Mastery is the posterior mean. New topics start at 50 percent, which is maximum uncertainty.

Each problem gets a context vector with five specific features: the mastery gap on its weakest topic, the mean gap across its topics, difficulty match, days since I last touched those topics, and a bias term. Difficulty match is the auto adjustment. Below 40 percent mastery it scores Easy problems higher. Above 70 percent it wants Hard ones.

The bandit is Thompson sampling over a linear model of expected learning gain. The reward for an attempt is the mastery gain in the problem's weakest topic. Early on the posterior is wide, so sampling explores. As evidence arrives it concentrates, so sampling exploits. There is no epsilon to tune.

## Quickstart

```bash
pip install -e .
grind-bandit next
grind-bandit log 1 solved --minutes 25
grind-bandit stats
```

State lives in `~/.grind-bandit/state.json`. Set `GRIND_BANDIT_STATE` to move it.

## A real session

This scripted a user who solves 90 percent of array and string problems and 30 percent of DP, graph, and backtracking ones. Thirty attempts logged, then:

```
$ grind-bandit next --seed 3
#46 Permutations (Medium)
https://leetcode.com/problems/permutations/
Topics: array, backtracking
Why this one:
  * weakest topic here: backtracking (mastery 25%)
  * practiced today
  * attacking your biggest gap
```

It found the weak topic and attacked it. The mastery view after those 30 attempts:

<details>
<summary>stats output</summary>

```
Mastery heatmap (darker blocks mean stronger mastery):
+----------------------+----------------------+----------------------+----------------------+
| array                | backtracking         | bfs                  | binary-search        |
| ▓▓▓▓▓▓▓▓  65%      | ▒▒▒▒▒▒▒▒  25%      | ▓▓▓▓▓▓▓▓  67%      | ████████  80%      |
+----------------------+----------------------+----------------------+----------------------+
| bit-manipulation     | bst                  | design               | dfs                  |
| ▓▓▓▓▓▓▓▓  67%      | ████████  75%      | ████████  75%      | ████████  86%      |
+----------------------+----------------------+----------------------+----------------------+
| divide-conquer       | dp                   | graph                | greedy               |
| ████████  75%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  67%      | ▒▒▒▒▒▒▒▒  33%      |
+----------------------+----------------------+----------------------+----------------------+
| hash-table           | heap                 | intervals            | linked-list          |
| ████████  86%      | ████████  75%      | ▓▓▓▓▓▓▓▓  67%      | ████████  80%      |
+----------------------+----------------------+----------------------+----------------------+
| math                 | recursion            | shortest-path        | sliding-window       |
| ████████  80%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  50%      |
+----------------------+----------------------+----------------------+----------------------+
| stack                | string               | topological-sort     | tree                 |
| ████████  75%      | ████████  75%      | ▓▓▓▓▓▓▓▓  67%      | ████████  75%      |
+----------------------+----------------------+----------------------+
| trie                 | two-pointers         | union-find           |
| ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  67%      |
+----------------------+----------------------+----------------------+----------------------+

topic             mastery  tried   solved   last practiced
backtracking      ██░░░░░░░░ 25% 2       0%       today
greedy            ███░░░░░░░ 33% 1       0%       today
two-pointers      █████░░░░░ 50% 2       50%      today
dp                █████░░░░░ 50% 4       50%      today
array             ██████░░░░ 65% 15      67%      today
hash-table        ████████░░ 86% 5       100%     today
```
</details>

And the weekly review on the same history:

```
$ grind-bandit review --weeks 2
Weekly review: last 2 week(s)

week ending 2026-10-10: 30 attempts, 24 solved (80%), 895 min
  most improved: hash-table (50% -> 86%)
  weakest now: backtracking (25%)
```

## Commands

`next`: recommends one problem and explains why. `--seed` makes it reproducible.

`log`: records an attempt. Example: `grind-bandit log 322 failed --minutes 45`.

`stats`: mastery heatmap plus a per topic table with attempts, solve rate, and last practiced date.

`review`: weekly summary with attempts, solve rate, minutes, most improved topic, and weakest topic. It replays the log to rebuild past mastery, so the trends are honest.

`export`: dumps history. `--format csv` or `--format json`, `--out` picks the file.

`reset`: wipes everything. It asks first unless you pass `--yes`.

## Honest limitations

The bandit needs about 20 log attempts to get enough data to work properly. Before that the posterior is mostly prior and the picks are close to random. That is by design, but expected.

Forgetting is not modeled. If you ignore graphs for a month, your graph mastery stays where it was. Spaced repetition is on the roadmap, not in the code.

The problem bank is 122 hand picked classics, not all of LeetCode. Coverage is good for interview prep.
The reward is mastery gain, which favors easy wins early. If you only log Easy solves, the bandit will keep feeding you Easy problems until the difficulty match feature catches up. Log honestly.

## License

MIT. See LICENSE.
