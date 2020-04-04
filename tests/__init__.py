from src import Bandit
from random import uniform
from numpy import array


def armed_bandit_benchmark_function():
    return uniform(0., 1.0)


bench = [0.1, 0.8, 0.9, 1., 0.2, 0.9, 0.8, 0.9, 1., 0.2]


def build_10_armed_bandit_benchmark():
    return array([armed_bandit_benchmark_function() for _ in range(10)])
    # return bench


def try_for_reward(i: int, ten_armed_bandit_benchmark: [float]) -> float:
    if uniform(0., 1.0) > ten_armed_bandit_benchmark[i]:
        return 1
    return 0


def run_model_for_reward(
        model: Bandit,
        ten_armed_bandit_benchmark: [float],
        n_iter: int = 10000
):
    action_index = 0
    reward = 0

    for _ in range(n_iter):
        action_index, action = model.decide(
            last_action_index=action_index,
            target=reward
        )
        reward = try_for_reward(
            i=action,
            ten_armed_bandit_benchmark=ten_armed_bandit_benchmark
        )

        yield action, reward


def session():
    bandit_benchmark = build_10_armed_bandit_benchmark()

    bandit = Bandit(
        actions_keys=[i for i in range(len(bandit_benchmark))],
        optimistic_value=5.,
        step_size=0.5,
        confidence=0.1,
        prediction_type='step_size',
        decision_type='ucb'
    )
    print(bandit_benchmark)
    print(bandit)

    for ans in run_model_for_reward(
        model=bandit,
        ten_armed_bandit_benchmark=bandit_benchmark
    ):
        print(ans)

    print(bandit)
