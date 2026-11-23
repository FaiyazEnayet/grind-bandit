"""Unit tests for the shadow skill model (GB15).

Known-answer fixtures: exact probabilities and scores, hand-computed
hierarchical shrinkage, a numeric-gradient check of the Rasch fit,
and controlled simulation with a known ground truth used only to
validate the fitting machinery. Simulated rows are never presented
as real practice data.
"""

import math

import pytest

from grindbandit.research import calibration as cal
from grindbandit.research import skill_model as sm

T0 = 1_700_000_000.0
DAY = 86400.0


def _obs(problem_id=1, difficulty="Easy", topics=("array",), solved=True, i=0):
    return {"problem_id": problem_id, "difficulty": difficulty,
            "topics": list(topics), "solved": solved,
            "ts": T0 + i * DAY, "attempt_index": i,
            "prior_attempts_on_problem": 0, "repeat": False}


def test_sigmoid_known_values():
    assert sm.sigmoid(0.0) == pytest.approx(0.5)
    assert sm.sigmoid(1.0) == pytest.approx(0.7310585786300049)
    assert sm.sigmoid(-1.0) == pytest.approx(0.2689414213699951)
    assert sm.sigmoid(20.0) == pytest.approx(1.0, abs=1e-8)
    assert sm.sigmoid(-20.0) == pytest.approx(0.0, abs=1e-8)


def test_brier_and_log_loss_known_values():
    # Coin-flip predictions on one success and one failure.
    assert cal.brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)
    assert cal.log_loss([0.5, 0.5], [True, False]) == pytest.approx(math.log(2))
    # Perfect predictions score zero.
    assert cal.brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert cal.log_loss([1.0, 0.0], [True, False]) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError):
        cal.brier_score([], [])


def test_calibration_bins_known_values():
    preds = [0.05, 0.15, 0.85, 0.95]
    outcomes = [False, True, True, True]
    bins = cal.calibration_bins(preds, outcomes)
    by_low = {b["bin_low"]: b for b in bins}
    assert by_low[0.0]["n"] == 1
    assert by_low[0.0]["observed_rate"] == pytest.approx(0.0)
    assert by_low[0.1]["n"] == 1
    assert by_low[0.1]["observed_rate"] == pytest.approx(1.0)
    assert by_low[0.8]["n"] == 1
    assert by_low[0.9]["n"] == 1
    assert by_low[0.9]["mean_predicted"] == pytest.approx(0.95)


def test_breakdown_by_difficulty_known_values():
    obs = [_obs(difficulty="Easy", solved=True, i=0),
           _obs(difficulty="Easy", solved=False, i=1),
           _obs(difficulty="Hard", solved=False, i=2)]
    rows = cal.breakdown_by_difficulty([0.5, 0.5, 0.5], obs)
    by_diff = {r["difficulty"]: r for r in rows}
    assert by_diff["Easy"]["n"] == 2
    assert by_diff["Easy"]["observed_rate"] == pytest.approx(0.5)
    assert by_diff["Easy"]["brier"] == pytest.approx(0.25)
    assert by_diff["Hard"]["n"] == 1
    assert by_diff["Hard"]["observed_rate"] == pytest.approx(0.0)


def test_breakdown_by_topic_counts_multi_topic_rows_under_each_topic():
    obs = [_obs(topics=("array", "hash-table"), solved=True, i=0),
           _obs(topics=("array",), solved=False, i=1)]
    rows = cal.breakdown_by_topic([0.5, 0.5], obs)
    by_topic = {r["topic"]: r for r in rows}
    assert by_topic["array"]["n"] == 2
    assert by_topic["hash-table"]["n"] == 1
    assert by_topic["hash-table"]["observed_rate"] == pytest.approx(1.0)


def test_global_rate_fit_known_value():
    train = [_obs(solved=True, i=i) for i in range(3)]
    train += [_obs(solved=False, i=i + 3) for i in range(1)]
    model = sm.fit_model("global_rate", train)
    assert model["rate"] == pytest.approx(0.75)
    assert sm.predict(model, [_obs(i=9)]) == [pytest.approx(0.75)]


def test_difficulty_heuristic_fit_known_values():
    train = [_obs(difficulty="Easy", solved=True, i=i) for i in range(4)]
    train += [_obs(difficulty="Easy", solved=False, i=4)]
    train += [_obs(difficulty="Hard", solved=False, i=i + 5) for i in range(2)]
    model = sm.fit_model("difficulty_heuristic", train)
    assert model["rates"]["Easy"] == pytest.approx(0.8)
    assert model["rates"]["Hard"] == pytest.approx(0.0)
    # An unseen difficulty falls back to the global rate (4/7).
    assert sm.predict(model, [_obs(difficulty="Medium", i=9)])[0] == pytest.approx(4 / 7)


def test_topic_mean_fit_known_values():
    train = [_obs(topics=("array",), solved=True, i=i) for i in range(3)]
    train += [_obs(topics=("array",), solved=False, i=3)]
    train += [_obs(topics=("dp",), solved=False, i=i + 4) for i in range(2)]
    model = sm.fit_model("topic_mean", train)
    # array: Beta(1+3, 1+1) mean 4/6; dp: Beta(1+0, 1+2) mean 1/4.
    assert model["topic_means"]["array"] == pytest.approx(4 / 6)
    assert model["topic_means"]["dp"] == pytest.approx(1 / 4)
    # A problem on both topics averages the two means.
    assert sm.predict(model, [_obs(topics=("array", "dp"), i=9)])[0] == pytest.approx(
        (4 / 6 + 1 / 4) / 2)
    # An unseen topic falls back to 0.5, the untouched-topic default.
    assert sm.predict(model, [_obs(topics=("graph",), i=9)])[0] == pytest.approx(0.5)


def test_hierarchical_beta_shrinks_toward_global_rate():
    # One topic with a single success, one with ten attempts at 50 percent.
    train = [_obs(topics=("rare",), solved=True, i=0)]
    train += [_obs(topics=("common",), solved=(i % 2 == 0), i=i + 1)
              for i in range(10)]
    model = sm.fit_model("hierarchical_beta", train)
    assert model["kind"] == "hierarchical_beta"
    assert model["kappa"] == sm.KAPPA
    # The global rate is 6/11; the rare topic's posterior mean sits
    # between its raw 1.0 and the global rate, pulled by shrinkage.
    mu = 6 / 11
    rare = model["topic_means"]["rare"]
    assert mu < rare < 1.0
    assert rare == pytest.approx((sm.KAPPA * mu + 1) / (sm.KAPPA + 1))
    # The well-evidenced topic stays close to its own 0.5.
    assert abs(model["topic_means"]["common"] - 0.5) < 0.05
    assert model["pathological"] == []


def test_hierarchical_beta_no_topic_variation_predicts_global_rate():
    # Every topic solves at the same rate: no between-topic variation,
    # so every shrunk topic mean sits at the global rate.
    train = []
    for i, topic in enumerate(["array", "dp", "graph", "string"]):
        train += [_obs(topics=(topic,), solved=(j % 2 == 0), i=4 * i + j)
                  for j in range(4)]
    model = sm.fit_model("hierarchical_beta", train)
    assert model["global_rate"] == pytest.approx(0.5)
    assert all(mean == pytest.approx(0.5)
               for mean in model["topic_means"].values())
    assert sm.predict(model, [_obs(topics=("array",), i=99)])[0] == pytest.approx(0.5)
    assert model["pathological"] == []


def test_rasch_fit_recovers_known_truth():
    obs = sm.simulate_observations(
        600,
        truth={"ability": 0.0,
               "topic_ability": {"array": 1.2, "dp": -1.2},
               "difficulty_offset": {"Easy": -1.0, "Medium": 0.0, "Hard": 1.0}},
        seed=7)
    train, _ = sm.split_holdout(obs)
    model = sm.fit_model("rasch", train)
    assert model["kind"] == "rasch"
    assert model["converged"] is True
    assert model["pathological"] == []
    # Difficulty offsets recover the truth within a generous tolerance;
    # Medium is anchored at zero by construction. The Hard tolerance
    # is wider: only 10 of 218 bank problems are Hard and half of
    # those are dp, so the Hard offset and the dp ability are partly
    # confounded and the fit splits the effect between them.
    assert abs(model["difficulty_offset"]["Easy"] - (-1.0)) < 0.35
    assert abs(model["difficulty_offset"]["Hard"] - 1.0) < 0.5
    assert model["difficulty_offset"]["Medium"] == pytest.approx(0.0)
    # Topic abilities recover the truth within tolerance.
    assert abs(model["topic_ability"]["array"] - 1.2) < 0.4
    assert abs(model["topic_ability"]["dp"] - (-1.2)) < 0.4


def test_rasch_gradient_matches_numeric_gradient():
    # Independent oracle: the analytic gradient of the penalized
    # negative log-likelihood must match central finite differences.
    obs = sm.simulate_observations(
        120,
        truth={"ability": 0.3,
               "topic_ability": {"array": 0.5},
               "difficulty_offset": {"Easy": -0.5, "Medium": 0.0, "Hard": 0.5}},
        seed=3)
    params, topics, _ = sm._rasch_pack(obs, {"ability": 0.3,
                                             "topic_ability": {"array": 0.5},
                                             "difficulty_offset": {"Easy": -0.5, "Medium": 0.0,
                                                                   "Hard": 0.5}})
    value, grad = sm._rasch_objective_and_grad(params, obs, topics)
    eps = 1e-6
    for i in range(len(params)):
        plus = params.copy()
        plus[i] += eps
        minus = params.copy()
        minus[i] -= eps
        numeric = (sm._rasch_objective_and_grad(plus, obs, topics)[0]
                   - sm._rasch_objective_and_grad(minus, obs, topics)[0]) / (2 * eps)
        assert grad[i] == pytest.approx(numeric, abs=1e-4), i


def test_rasch_predictions_are_monotone_in_difficulty():
    # With a fitted model, a Hard problem must score below a Medium one
    # and an Easy one above it, holding the topic fixed: difficulty
    # separates from ability instead of loading onto it.
    obs = sm.simulate_observations(
        400,
        truth={"ability": 0.0, "topic_ability": {},
               "difficulty_offset": {"Easy": -1.0, "Medium": 0.0, "Hard": 1.0}},
        seed=5)
    train, _ = sm.split_holdout(obs)
    model = sm.fit_model("rasch", train)
    base = {"problem_id": 1, "topics": ["array"], "solved": True,
            "ts": T0, "attempt_index": 0,
            "prior_attempts_on_problem": 0, "repeat": False}
    preds = sm.predict(model, [{**base, "difficulty": d}
                               for d in ("Easy", "Medium", "Hard")])
    assert preds[0] > preds[1] > preds[2]


def test_extract_observations_counts_repeats():
    from grindbandit.research import evaluate_skill as ev

    state = {"log": [
        {"problem_id": 1, "difficulty": "Easy", "topics": ["array"],
         "solved": True, "ts": T0},
        {"problem_id": 1, "difficulty": "Easy", "topics": ["array"],
         "solved": False, "ts": T0 + DAY},
        {"problem_id": 2, "difficulty": "Hard", "topics": ["dp"],
         "solved": False, "ts": T0 + 2 * DAY},
    ]}
    extracted = ev.extract_observations(state)
    assert extracted["n_observations"] == 3
    assert extracted["n_repeats"] == 1
    second = extracted["observations"][1]
    assert second["repeat"] is True
    assert second["prior_attempts_on_problem"] == 1
    assert extracted["observations"][0]["repeat"] is False


def test_fit_model_rejects_empty_train():
    with pytest.raises(ValueError):
        sm.fit_model("global_rate", [])
    with pytest.raises(ValueError):
        sm.fit_model("rasch", [])
    with pytest.raises(ValueError):
        sm.fit_model("unknown_kind", [_obs()])


def test_all_solved_train_is_pathological():
    obs = [_obs(solved=True, i=i) for i in range(80)]
    train, holdout = sm.split_holdout(obs)
    report = sm.compare_models(train, holdout)
    assert report["models"]["rasch"]["pathological"], report["models"]["rasch"]
    assert "degenerate_train_all_solved" in report["models"]["rasch"]["pathological"]
    assert report["models"]["hierarchical_beta"]["pathological"]
    assert report["verdict"] in ("KEEP_SIMPLE", "NOT_IDENTIFIABLE")
