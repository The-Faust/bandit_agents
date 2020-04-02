from numpy import zeros, full

from src.model.functions import action_decision_functions


# TODO: find a way so that epsilon, confidence and step_size
#  are not stored in the object directly since they are not always needed
#  (they depend on the decision function employed by the peson using the model)
class BaseBandit:
    def __init__(
            self, actions_keys: [any],
            optimistic_value=0, step_size=0.01, epsilon=0.01, confidence=0.01
    ) -> None:
        self._step_size = step_size
        self._epsilon = epsilon
        self._confidence = confidence
        self._actions = actions_keys
        self._actions_speculated_rewards = full(len(actions_keys), optimistic_value)
        self._actions_counts = zeros(len(actions_keys))
        self._prediction_decision_functions = action_decision_functions

    def set_step_size(self, step_size_value: int) -> None:
        self._step_size = step_size_value

    def reset_action_count(self):
        self._actions_counts.fill(0)

    def set_action_speculated_rewards(self, rewards_value: float):
        self._actions_speculated_rewards.fill(rewards_value)

    def reset_action_specualed_rewards(self):
        self._actions_speculated_rewards.fill(0)

    def _increment_action_count(self, index: int):
        self._actions_counts[index] += 1


