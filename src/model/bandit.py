from numpy import count_nonzero, nditer
from src.model.base_bandit import BaseBandit
from src.model.arguments import Arguments
from src.model.functions.bounds import bound_functions_dict
from src.model.functions.prediction import prediction_functions_dict
from src.model.functions.decision import decision_functions_dict


class Bandit(BaseBandit, Arguments):
    def __init__(
            self, actions_keys: [str],
            optimistic_value: float = 0.,
            step_size: float = 0.01,
            epsilon: float = 0.01,
            confidence: float = 0.01,
            bound_type: str = 'ucb',
            prediction_type: str = 'step_size',
            decision_type: str = 'e_greedy'
    ) -> None:
        BaseBandit.__init__(
            self,
            actions_keys=actions_keys,
            optimistic_value=optimistic_value,
            step_size=step_size,
            epsilon=epsilon,
            confidence=confidence,
            bound_type=bound_type,
            prediction_type=prediction_type,
            decision_type=decision_type
        )
        Arguments.__init__(self)

        self.bound_functions = bound_functions_dict
        self.prediction_functions = prediction_functions_dict
        self.decision_functions = decision_functions_dict

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

        next_action_index = self._choose_action()
        self._increment_action_count(next_action_index)
        return next_action_index, self._actions_keys[next_action_index]

    def _update_speculated_rewards_array(
            self, last_action_index: int, target: float
    ) -> [float]:
        return self.bound_functions[self._bound_type](
            self.prediction_functions[self._prediction_type](
                *self._get_prediction_function_arguments(
                    last_action_index=last_action_index,
                    target=target
                )
            ), *self._get_bound_function_arguments()
        )

    def _choose_action(self) -> int:
        return self.decision_functions[self._decision_type](
            *self._get_decision_function_arguments()
        )

    def __eq__(self, other):
        return isinstance(other, type(self)) and BaseBandit.__eq__(self, other)

    def __copy__(self):
        new_bandit = Bandit(actions_keys=self._actions_keys)
        new_bandit.__dict__ = self.__dict__
        return new_bandit
