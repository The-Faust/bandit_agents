from typing import Callable, List

from src.domain.types import action_key_type, bandit_key_type
from src.domain import ContextBandit


class ContextAction(object):
    def __init__(
            self, action_key: action_key_type,
            action: Callable[..., float],
            bandit_key_list: List[bandit_key_type] = ()
    ):
        self.action_key: action_key_type = action_key
        self.action: Callable[..., float] = action
        self.action_count: int = 0
        self.bandit_key_list: List[bandit_key_type] = bandit_key_list

    def add_bandit_key(self, bandit_key: bandit_key_type):
        self.bandit_key_list.append(bandit_key)

    def act(
            self, bandit_who_acted: bandit_key_type,
            bandit_dict: {bandit_key_type: ContextBandit},
            *args, **kwargs
    ) -> float:
        reward = self.action(*args, **kwargs)

        self.action_count += 1
        self.update_bandits(
            self.action_key,
            reward,
            bandit_who_acted,
            bandit_dict
        )

        return reward

    # TODO: watch for alternative when updating bandit. this will do for now,
    #  but will be refactored and reviewed in later releases
    def update_bandits(
            self, action_key: action_key_type,
            target: float,
            bandit_who_acted: bandit_key_type,
            bandit_dict: {bandit_key_type: ContextBandit}
    ) -> None:
        map(
            lambda bandit_key: bandit_dict[bandit_key].bandit.fit_with_action_key(action_key, target),
            filter(lambda bandit_key: bandit_key != bandit_who_acted, self.bandit_key_list)
        )

    def increment_count(self):
        self.action_count += 1

    def __str__(self):
        return 'ContextAction with action: {} \n ' \
               '    action taken {} times \n\n' \
            .format(self.action, self.action_count)

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.action == self.action
