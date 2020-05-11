from src.model.bandit import Bandit


class ContextBandit(object):
    def __init__(
            self, bandit: Bandit
    ):
        self.bandit = bandit
        self.last_action_index = 0

    def set_last_action_taken_index(self, last_action_index: int):
        self.last_action_index = last_action_index

    def __str__(self):
        return 'ContextBandit: \n' \
               '    Bandit is {} \n' \
               '    last action index is: {}'\
            .format(self.bandit, self.last_action_index)

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.bandit == self.bandit
