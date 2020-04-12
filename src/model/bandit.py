from numpy import count_nonzero, nditer
from src.model.base_bandit import BaseBandit
from src.model.arguments import Arguments


class Bandit(BaseBandit, Arguments):
    def __init__(
            self, actions_keys: [str],
            optimistic_value=0, step_size=0.01, epsilon=0.01, confidence=0.01,
            prediction_type='step_size', decision_type='e_greedy'
    ) -> None:
        BaseBandit.__init__(
            self,
            actions_keys=actions_keys,
            optimistic_value=optimistic_value,
            step_size=step_size,
            epsilon=epsilon,
            confidence=confidence,
            prediction_type=prediction_type,
            decision_type=decision_type
        )

    def fit(self, actions_indexes: [int], target_list: [float]):
        map(self.decide, nditer([actions_indexes, target_list]))

    def decide(
            self, last_action_index=-1, target=0
    ) -> (int, any):
        if count_nonzero(self._actions_counts) > 0:
            self._actions_speculated_rewards = self._update_speculated_rewards_array(
                last_action_index=last_action_index,
                target=target
            )

        next_action_index = self._choose_action(decision_type=self._decision_type)
        self._increment_action_count(next_action_index)
        return next_action_index, self._actions_keys[next_action_index]

    def _update_speculated_rewards_array(
            self, last_action_index: int, target: float
    ) -> [float]:
        return self._prediction_decision_functions.prediction_formulas.__dict__['step_size'](
            *self._get_prediction_function_arguments(
                prediction_type=self._prediction_type,
                last_action_index=last_action_index,
                target=target
            )
        )

    def _choose_action(self, decision_type='e_greedy') -> int:
        return self._prediction_decision_functions.decision_formulas.__dict__[decision_type](
            *self._get_decision_function_arguments(decision_type)
        )

    def __eq__(self, other):
        return isinstance(other, type(self)) and BaseBandit.__eq__(self, other)

    def __copy__(self):
        new_bandit = Bandit(actions_keys=self._actions_keys)
        new_bandit.__dict__ = self.__dict__
        return new_bandit
