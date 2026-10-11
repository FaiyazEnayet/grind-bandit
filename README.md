# grind-bandit

Your LeetCode grind as a contextual bandit.

I used to grind problems at random. Arrays on Monday, whatever felt easy on Tuesday. My DP stayed bad for months. So I built the thing that picks my next problem for me. It explores my weak topics and exploits what is almost mastered. That tradeoff is handled by Bayes, not by vibes.

## The math

Each topic gets a Beta Bernoulli skill model. A solve adds a win, a fail adds a loss. Mastery is the posterior mean. New topics start at 50 percent, which is maximum uncertainty.

Each problem gets a context vector with five features: the mastery gap on its weakest topic, the mean gap across its topics, difficulty match, days since I last touched those topics, and a bias term. Difficulty match is the auto adjustment. Below 40 percent mastery it scores Easy problems higher. Above 70 percent it wants Hard ones.

The bandit is Thompson sampling over a linear model of expected learning gain. The reward for an attempt is the mastery gain in the problem's weakest topic. Early on the posterior is wide, so sampling explores. As evidence arrives it concentrates, so sampling exploits. There is no epsilon to tune.

Mastery forgets. Every read decays the stored win and loss counts exponentially toward the prior, based on days since the topic was last attempted. The half-life is 30 days. That number is my choice, not a measured fact about memory. You can change it in the state file under config.half_life_days. The stored counts never change, so the raw evidence stays intact. Stale topics sink back toward 50 percent and the bandit starts recommending them again.

## Quickstart

```bash
pip install -e .
grind-bandit next
grind-bandit log 1 solved --minutes 25
grind-bandit stats
```

State lives in `~/.grind-bandit/state.json`. Set `GRIND_BANDIT_STATE` to move it.

## A real session

I scripted a user who solves 90 percent of array and string problems and 30 percent of DP, graph, and backtracking ones. Thirty attempts logged, then:

```
$ grind-bandit next --seed 3
#25 Reverse Nodes in k-Group (Hard)
https://leetcode.com/problems/reverse-nodes-in-k-group/
Topics: linked-list
Why this one:
  * weakest topic here: linked-list (mastery 34%)
  * last touched 3 days ago
  * broad room to grow across these topics
```

It found the weak topic and attacked it. The mastery view after those 30 attempts:

<details>
<summary>stats output</summary>

```
Mastery heatmap (darker blocks mean stronger mastery):
+----------------------+----------------------+----------------------+----------------------+
| array                | backtracking         | bfs                  | binary-search        |
| ▓▓▓▓▓▓▓▓  72%      | ▓▓▓▓▓▓▓▓  50%      | ▒▒▒▒▒▒▒▒  34%      | ████████  85%      |
+----------------------+----------------------+----------------------+----------------------+
| bit-manipulation     | bst                  | design               | dfs                  |
| ▒▒▒▒▒▒▒▒  35%      | ▓▓▓▓▓▓▓▓  64%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  60%      |
+----------------------+----------------------+----------------------+----------------------+
| divide-conquer       | dp                   | graph                | greedy               |
| ▓▓▓▓▓▓▓▓  72%      | ░░░░░░░░  17%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  50%      |
+----------------------+----------------------+----------------------+----------------------+
| hash-table           | heap                 | intervals            | linked-list          |
| ▓▓▓▓▓▓▓▓  74%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  50%      | ▒▒▒▒▒▒▒▒  34%      |
+----------------------+----------------------+----------------------+----------------------+
| math                 | recursion            | shortest-path        | sliding-window       |
| ████████  84%      | ▓▓▓▓▓▓▓▓  64%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  63%      |
+----------------------+----------------------+----------------------+----------------------+
| stack                | string               | topological-sort     | tree                 |
| ▓▓▓▓▓▓▓▓  60%      | ████████  83%      | ▓▓▓▓▓▓▓▓  50%      | ▓▓▓▓▓▓▓▓  66%      |
+----------------------+----------------------+----------------------+----------------------+
| trie                 | two-pointers         | union-find           |
| ▓▓▓▓▓▓▓▓  74%      | ████████  80%      | ▒▒▒▒▒▒▒▒  37%      |
+----------------------+----------------------+----------------------+----------------------+

topic             mastery  tried   solved   last practiced
dp                █░░░░░░░░░ 17% 4       0%       today
linked-list       ███░░░░░░░ 34% 1       0%       3d ago
bfs               ███░░░░░░░ 34% 1       0%       4d ago
bit-manipulation  ███░░░░░░░ 35% 1       0%       6d ago
union-find        ███░░░░░░░ 37% 1       0%       13d ago
design            █████░░░░░ 50% 2       50%      11d ago
backtracking      █████░░░░░ 50% 0       0%       never
graph             █████░░░░░ 50% 0       0%       never
greedy            █████░░░░░ 50% 0       0%       never
heap              █████░░░░░ 50% 0       0%       never
intervals         █████░░░░░ 50% 0       0%       never
shortest-path     █████░░░░░ 50% 0       0%       never
topological-sort  █████░░░░░ 50% 0       0%       never
dfs               █████░░░░░ 60% 3       67%      3d ago
stack             █████░░░░░ 60% 3       67%      2d ago
sliding-window    ██████░░░░ 63% 1       100%     14d ago
bst               ██████░░░░ 64% 1       100%     10d ago
recursion         ██████░░░░ 64% 1       100%     9d ago
tree              ██████░░░░ 66% 4       75%      3d ago
divide-conquer    ███████░░░ 72% 2       100%     10d ago
array             ███████░░░ 72% 16      75%      today
hash-table        ███████░░░ 74% 6       83%      8d ago
trie              ███████░░░ 74% 2       100%     4d ago
two-pointers      ███████░░░ 80% 3       100%     1d ago
string            ████████░░ 83% 4       100%     1d ago
math              ████████░░ 84% 5       100%     7d ago
binary-search     ████████░░ 85% 5       100%     1d ago
```
</details>

And the weekly review on the same history:

```
$ grind-bandit review --weeks 2
Weekly review: last 2 week(s)

week ending 2026-10-11: 13 attempts, 8 solved (62%), 383 min
  most improved: binary-search (74% -> 85%)
  weakest now: dp (17%)

week ending 2026-10-04: 14 attempts, 12 solved (86%), 369 min
  most improved: tree (50% -> 75%)
  weakest now: union-find (35%)
```

## Commands

`next`: recommends one problem and explains why. `--seed` makes it reproducible.

`log`: records an attempt. Example: `grind-bandit log 322 failed --minutes 45`.

`stats`: mastery heatmap plus a per topic table with attempts, solve rate, and last practiced date.

`review`: weekly summary with attempts, solve rate, minutes, most improved topic, and weakest topic. It replays the log to rebuild past mastery, so the trends are honest.

`export`: dumps history. `--format csv` or `--format json`, `--out` picks the file.

`reset`: wipes everything. It asks first unless you pass `--yes`.

## Honest limitations

The bandit is dumb until you log about 20 attempts. Before that the posterior is mostly prior and the picks are close to random. That is by design, but expect it.

The problem bank is 218 hand picked classics, not all of LeetCode. Coverage is good for interview prep and thin everywhere else.

The reward is mastery gain, which favors easy wins early. If you only log Easy solves, the bandit will keep feeding you Easy problems until the difficulty match feature catches up. Log honestly.

## License

MIT. See LICENSE.
