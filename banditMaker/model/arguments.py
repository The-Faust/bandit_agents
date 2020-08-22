from typing import Sequence, Iterable

"""
This is a mixin class so omission of class parameters are intended
(said parameters must exists in child class)
"""


# TODO: these should not get too big but maybe change the way information is accessed?
class Arguments:
    def __init__(self):
        self.prediction_arguments_dict = {
            'step_size': self._step_size,
            'mean': self._actions_counts
        }

        self.decision_arguments_dict = {
            'e_greedy':
                self._epsilon,
            'greedy': 0
        }

        self.bound_arguments_dict = {
            'ucb': (self._actions_counts, self._confidence),
            'null': ()
        }

    def _get_prediction_function_arguments(
            self, last_action_index: int, target: float,
    ) -> Sequence[float]:
        arguments = (
            last_action_index,
            target,
            self.prediction_arguments_dict[self._prediction_type]
        )
        return (
            self._actions_speculated_rewards,
            *arguments
        )

    def _get_decision_function_arguments(self) -> Sequence[float]:
        return (
            self._actions_speculated_rewards,
            self.decision_arguments_dict[self._decision_type]
        )

    def _get_bound_function_arguments(self) -> Sequence:
        return self.bound_arguments_dict[self._bound_type]
