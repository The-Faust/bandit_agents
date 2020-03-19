from numpy import zeros, full
from static_bandit import action_decision_functions


class BaseBandit:
    def __init__(self, actions: [any], optimistic_value=0, step_size=0.01) -> None:
        self.step_size = step_size
        self.actions_speculated_rewards = full(len(actions), optimistic_value)
        self.actions_counts = zeros(len(actions))
        self.action_decision_functions = action_decision_functions