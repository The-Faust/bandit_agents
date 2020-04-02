from numpy import sum, count_nonzero, nditer
from src.model import BaseBandit


class Bandit(BaseBandit):
    def __init__(
            self, actions_keys: [any],
            optimistic_value=0, step_size=0.01, confidence=0.01
    ) -> None:
        BaseBandit.__init__(
            self, actions_keys,
            optimistic_value, step_size, confidence
        )

    def fit(
            self, predictions_types: [str], decisions_types: [str], actions_indexes: [int], target_list: [float]
    ):
        for i, j, k, l in nditer([predictions_types, decisions_types, actions_indexes, target_list]):
            self.decide(i, j, k, l)

    def decide(
            self, prediction_type='step_size', decision_type='e_greedy', last_action_index=-1, target=0
    ) -> (int, any):
        if count_nonzero(self._actions_counts):
            self._update_speculated_rewards_array(
                last_action_index=last_action_index,
                target=target,
                prediction_type=prediction_type
            )

        next_action_index = self._choose_action(decision_type=decision_type)
        self._increment_action_count(next_action_index)
        return next_action_index, self._actions[next_action_index]

    def _update_speculated_rewards_array(
            self, last_action_index: int, target: float, prediction_type='step_size'
    ) -> [float]:
        return self._prediction_decision_functions.prediction_formulas.__dict__[prediction_type](
            *self._get_prediction_function_arguments(
                prediction_type=prediction_type,
                last_action_index=last_action_index,
                target=target
            )
        )

    def _choose_action(self, decision_type='e_greedy') -> int:
        print(self._prediction_decision_functions.__class__.__dict__.keys())
        return self._prediction_decision_functions.decision_formulas.__dict__[decision_type](
            *self._get_decision_function_arguments(decision_type=decision_type)
        )

    def _get_prediction_function_arguments(
            self, prediction_type: str, last_action_index: int, target: float
    ) -> tuple:
        return {
            'step_size': (
                self._actions_speculated_rewards,
                last_action_index,
                target,
                self._step_size
            ),
            'mean': (
                self._actions_speculated_rewards,
                last_action_index,
                target,
                self._actions_counts
            )
        }[prediction_type]

    def _get_decision_function_arguments(self, decision_type: str) -> tuple:
        return {
            'e_greedy': (
                self._epsilon,
                self._actions_speculated_rewards
            ),
            'ucb': (
                self._actions_speculated_rewards,
                self._actions_counts,
                self._confidence
            )
        }[decision_type]
