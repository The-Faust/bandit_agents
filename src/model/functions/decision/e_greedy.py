from numpy import argmax
from random import uniform, randint


def e_greedy(speculated_reward_array: [float], epsilon: float) -> int:
    return _e_greedy_chose_action(speculated_reward_array, _is_greedy(epsilon))


def _e_greedy_chose_action(speculated_rewards_array: [float], is_greedy: bool) -> int:
    return argmax(
        speculated_rewards_array
    ) if is_greedy else randint(
        0, len(speculated_rewards_array) - 1
    )


def _is_greedy(epsilon: float) -> bool:
    return uniform(0.0, 1.0) > epsilon
