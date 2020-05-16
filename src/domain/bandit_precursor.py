from typing import List
from src.domain.types import action_key_type, bandit_key_type


class BanditPrecursor(object):
    def __init__(
            self, actions_keys: List[action_key_type],
            bandit_key: bandit_key_type,
            optimistic_value: float = 0.0,
            step_size: float = 0.01,
            epsilon: float = 0.01,
            confidence: float = 0.01,
            bound_type: str = 'ucb',
            prediction_type: str = 'step_size',
            decision_type: str = 'e_greedy'
    ):
        self.bandit_key = bandit_key
        self.actions = actions_keys
        self.optimistic_value = optimistic_value
        self.step_size = step_size
        self.epsilon = epsilon
        self.confidence = confidence
        self.bound_type = bound_type
        self.prediction_type = prediction_type
        self.decision_type = decision_type

    def get_bandit_arguments(self) -> (
            [any], float, float, float, float, str, str
    ):
        return self.actions, \
               self.optimistic_value, \
               self.step_size, \
               self.epsilon, \
               self.confidence, \
               self.bound_type, \
               self.prediction_type, \
               self.decision_type
