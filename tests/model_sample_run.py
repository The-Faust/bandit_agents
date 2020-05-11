from src.model.bandit import Bandit
from random import uniform
from numpy import array


def armed_bandit_benchmark_function():
    return uniform(0., 1.0)


def build_10_armed_bandit_benchmark():
    return array([armed_bandit_benchmark_function() for _ in range(10)])


# This is effectively a bandit-machine with Bernoulli functions
def try_for_reward(index: int, ten_armed_bandit_benchmark: [float]) -> float:
    if uniform(0., 1.0) > ten_armed_bandit_benchmark[index]:
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
            index=action,
            ten_armed_bandit_benchmark=ten_armed_bandit_benchmark
        )

        yield action, reward


def session(n_iter=10000):
    bandit_benchmark = build_10_armed_bandit_benchmark()

    bandit = Bandit(
        actions_keys=[i for i in range(len(bandit_benchmark))],
        optimistic_value=5.,
        step_size=0.5,
        confidence=0.1,
        prediction_type='step_size',
        decision_type='greedy'
    )
    print(bandit)
    print(bandit_benchmark)

    for ans in run_model_for_reward(
        model=bandit,
        ten_armed_bandit_benchmark=bandit_benchmark,
        n_iter=n_iter
    ):
        continue

    print(bandit)
