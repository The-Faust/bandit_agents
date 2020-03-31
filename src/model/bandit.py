from src.model import BaseBandit


class Bandit(BaseBandit):
    def __init__(
            self, n_of_actions: [any],
            optimistic_value=0, step_size=0.01,
            prediction_type='step_size', decision_type='e_greedy'
    ) -> None:
        BaseBandit.__init__(
            self, n_of_actions,
            optimistic_value, step_size,
            prediction_type, decision_type
        )

    def predict(self, prediction_type='step_size') -> (int, float):

        action_index = self._action_decision_functions.decision_formulas.__dict__[prediction_type](
            speculated_reward_array=self._actions_speculated_rewards
        )
        return action_index, self._actions_speculated_rewards[action_index]

    def fit(self, reward, ):
        pass

    def _update_speculated_rewards_array(self, action_index: int, target: float, weight_type='mean') -> None:
        self._actions_speculated_rewards[action_index] = self._action_decision_functions.prediction_formulas.__dict__[
            weight_type
        ](
            self._actions_speculated_rewards[action_index],
            target,
            self._step_size
            if weight_type == 'step_size'
            else self._actions_counts[action_index]
        )

    def _choose_action(self) -> int:
        return self._action_decision_functions.decision_formulas.__dict__[
            self._decision_type
        ].make_decision(self._actions_speculated_rewards)

    def _get_decision_function_arguments(self, decision_type: str) -> tuple:
        return {
            'e_greedy': (self._epsilon, self._actions_speculated_rewards),
            'ucb': (self._actions_speculated_rewards, self._actions_counts, self._confidence)
        }[decision_type]
