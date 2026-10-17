<p align="center"><img src="docs/readme-banner.svg" alt="Grind-Bandit: Practice recommendations with visible uncertainty" width="1200"></p>

[![Tests](https://github.com/FaiyazEnayet/grind-bandit/actions/workflows/ci.yml/badge.svg)](https://github.com/FaiyazEnayet/grind-bandit/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-007EC6)](LICENSE) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB) ![Built with NumPy + contextual bandits](https://img.shields.io/badge/built%20with-NumPy%20%2B%20contextual%20bandits-555555) [![Interactive demo](https://img.shields.io/badge/demo-interactive-4C9D39)](https://faiyazenayet.github.io/grind-bandit/docs/dashboard.html)

<p align="center"><a href="https://faiyazenayet.github.io/grind-bandit/docs/dashboard.html"><b>Open interactive demo</b></a> · <a href="docs/dashboard.html">HTML file</a> · <a href="#method">Method</a> · <a href="#run-locally">Run locally</a> · <a href="#limitations">Limitations</a></p>

# Grind-Bandit

Grind-Bandit recommends programming practice from logged attempts. It models topic mastery and forgetting, then explains why a particular problem is a useful next step.

**Python · contextual bandits · learning-state models**

[![Grind-Bandit demo](docs/demo.gif)](https://faiyazenayet.github.io/grind-bandit/docs/dashboard.html)


## Explore

- Compare estimated mastery across 27 topics and inspect the evidence behind a recommendation.
- Switch recommendation samples and explore topic-level forgetting trajectories.
- Keep observed solve rate separate from model-estimated mastery.

## Method

Attempt history updates topic estimates and the bandit state. Recommendations combine estimated learning gain, uncertainty and difficulty fit. The dashboard exposes a reproducible scripted history rather than implying it is a real student study.

## Evidence at a glance

| Scope | Recorded result or contract |
| :--- | :--- |
| Problem bank | 218 hand-picked problems across 27 topics |
| Dashboard history | 30 scripted attempts with seed 42 |
| Interpretation | A reproducible demonstration, not a measured learning-outcomes trial |

## Run locally


```sh
python -m pip install -e .
python -m pip install -r portfolio_ui/requirements.txt
python -m pytest tests -q
python -m grindbandit --help
python scripts/make_dashboard.py
```

## Technical design

The learner state separates problem attempts, topic estimates, logs and bandit parameters. Recommendations account for uncertainty and difficulty fit as well as expected learning gain. Forgetting reduces estimated mastery between practice events, so an old success does not imply permanently mastered knowledge.

```mermaid
flowchart LR
  N0["Attempt history"] --> N1["Topic and forgetting state"]
  N0["Attempt history"] --> N2["Bandit context"]
  N1["Topic and forgetting state"] --> N3["Recommendation"]
  N2["Bandit context"] --> N3["Recommendation"]
  N3["Recommendation"] --> N4["Explained next problem"]
```

## Evaluation protocol

The committed demonstration scripts a learner with different success tendencies across topics. That is useful for checking model behavior and reproducibility, but it is not a human-subject learning experiment. Recommendation changes should be checked against simple practice policies before being called better pedagogy.

![How a modeled solve or failure changes the estimate](docs/results.png)

A high topic estimate is a model belief informed by the logged history. The displayed solve rate is an observed fraction within the demonstration. They should not be equated, and uncertainty matters most during cold start and in sparsely attempted topics.

## Code and data

`src/grindbandit/bandit.py` owns recommendations. `mastery.py` handles learning and forgetting estimates. `state.py` stores attempts. The source problem bank and dashboard scripts remain inspectable.


## Limitations

- Cold-start recommendations depend heavily on priors.
- The forgetting half-life and difficulty thresholds are modeling choices.
- Easy-only histories can bias estimated learning gain. State saving also needs an atomic-write and relative-path fix.

## License

[MIT](LICENSE). Third-party datasets, libraries and fonts retain their own terms.
