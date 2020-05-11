from typing import Callable


class ContextAction(object):
    def __init__(
            self, action: Callable[..., float]
    ):
        self.action = action
        self.action_count: int = 0

    def increment_count(self):
        self.action_count += 1

    def act(self, *args, **kwargs):
        reward = self.action(*args, **kwargs)
        self.action_count += 1
        return reward

    def __str__(self):
        return 'ContextAction with action: {} \n ' \
               '    action taken {} times \n\n'\
            .format(self.action, self.action_count)

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.action == self.action
