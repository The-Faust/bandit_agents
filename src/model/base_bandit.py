from numpy import zeros, full, array

from src.domain.types import action_key_type, bandit_key_type


# TODO: find a way so that epsilon, confidence and step_size
#  are not stored in the object directly since they are not always needed
#  (they depend on the decision function employed by the person using the model)
class BaseBandit(object):
    def __init__(
            self, actions_keys: [action_key_type],
            optimistic_value: float = 0.,
            step_size: float = 0.01,
            epsilon: float = 0.01,
            confidence: float = 0.01,
            bound_type: str = 'ucb',
            prediction_type: str = 'step_size',
            decision_type: str = 'greedy'
    ) -> None:
        self._step_size = step_size
        self._epsilon = epsilon
        self._confidence = confidence
        self._actions_keys = array(actions_keys)
        self._actions_speculated_rewards = full(len(actions_keys), optimistic_value)
        self._actions_counts = zeros(len(actions_keys))
        self._bound_type = bound_type
        self._prediction_type = prediction_type
        self._decision_type = decision_type

    def set_step_size(self, step_size_value: int) -> None:
        self._step_size = step_size_value

    def reset_action_count(self):
        self._actions_counts.fill(0)

    def reset_action_speculated_rewards(self):
        self._actions_speculated_rewards.fill(0)

    def set_bound_type(self, bound_type: str):
        self._bound_type = bound_type

    def set_prediction_type(self, prediction_type: str):
        self._prediction_type = prediction_type

    def set_decision_type(self, decision_type: str):
        self._decision_type = decision_type

    def _increment_action_count(self, index: int):
        self._actions_counts[index] += 1

    def __str__(self) -> str:
        return 'Multi-armed {} bandit model, using {} prediction method \n\n ' \
               '  with step_size {}, epsilon {}, confidence {} \n\n' \
               '  actions are: {} \n' \
               '  current biases are: {} \n' \
               '  actions counts are: {} \n\n '.format(
                    self._decision_type,
                    self._prediction_type,
                    self._step_size,
                    self._epsilon,
                    self._confidence,
                    self._actions_keys,
                    self._actions_speculated_rewards,
                    self._actions_counts
                )

    def __eq__(self, other) -> bool:
        return all([
            self._step_size == other._step_size,
            self._epsilon == other._epsilon,
            self._confidence == other._confidence,
            self._bound_type == other._bound_type,
            self._decision_type == other._decision_type,
            self._prediction_type == other._prediction_type,
            all(self._actions_keys == other._actions_keys),
            all(self._actions_speculated_rewards == other._actions_speculated_rewards),
            all(self._actions_counts == other._actions_counts)
        ])
