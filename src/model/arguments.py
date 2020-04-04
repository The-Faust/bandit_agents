"""
This is a mixin class so omission of class parameters are intended
(said parameters must exists in child class)
"""

class Arguments:
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