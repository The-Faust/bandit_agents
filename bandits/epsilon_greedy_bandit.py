from static_bandit.base_bandit import BaseBandit


class EpsilonGreedyBandit(BaseBandit):
    def __init__(self, actions: [any], optimistic_value=0, step_size=0.01):
        BaseBandit.__init__(self, actions, optimistic_value, step_size)
