from numpy import zeros, full

from src.model.functions import action_decision_functions


# TODO: find a way so that epsilon, confidence and step_size
#  are not stored in the object directly since they are not always needed
#  (they depend on the decision function employed by the peson using the model)
class BaseBandit:
    def __init__(
            self, n_of_actions: int,
            optimistic_value=0, step_size=0.01, epsilon=0.01, confidence=0.01,
            prediction_type='step_size', decision_type='e_greedy'
    ) -> None:
        self._step_size = step_size
        self._epsilon = epsilon
        self._confidence = confidence
        self._actions_speculated_rewards = full(n_of_actions, optimistic_value)
        self._actions_counts = zeros(n_of_actions)
        self._prediction_type = prediction_type
        self._decision_type = decision_type
        self._action_decision_functions = action_decision_functions

    def set_step_size(self, step_size_value: int) -> None:
        self._step_size = step_size_value

    def set_prediction_type(self, prediction_type: str) -> None:
        self._prediction_type = prediction_type

    def set_decision_type(self, decision_type: str) -> None:
        self._decision_type = decision_type

    def reset_action_count(self):
        self._actions_counts.fill(0)

    def set_action_speculated_rewards(self, rewards_value: float):
        self._actions_speculated_rewards.fill(rewards_value)

    def reset_action_specualed_rewards(self):
        self._actions_speculated_rewards.fill(0)
