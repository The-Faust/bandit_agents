from numpy import zeros, full
from static_bandit.base_bandit import BaseBandit


class GreedyBandit(BaseBandit):
    def __init__(self, actions: [any], optimistic_value=0, step_size=0.01) -> None:
        BaseBandit.__init__(self, actions, optimistic_value, step_size)

    def predict(self) -> (int, float):
        action_index = self.action_decision_functions.decision_formulas.greedy(
            speculated_reward_array=self.actions_speculated_rewards
        )
        return action_index, self.actions_speculated_rewards[action_index]

    def fit(self, reward, ):
        pass

    def _update_speculated_rewards_array(self, action_index: int, target: float, weight_type='mean') -> None:
        self.actions_speculated_rewards[action_index] = self.action_decision_functions.prediction_formulas[weight_type](
            last_prediction=self.actions_speculated_rewards[action_index],
            target=target,
            step_size=self.step_size
        )

    def _choose_action(self) -> int:
        return self.action_decision_functions.greedy.make_decision(self.actions_speculated_rewards)
