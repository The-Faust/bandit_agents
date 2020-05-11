from typing import Callable, Iterable

from src.context.base_context import BaseContext
from src.Shared.exceptions import ex, context_exceptions as exceptions

from typing import TypeVar, List

action_key_type = TypeVar('action_key_type', int, float, str)
bandit_key_type = TypeVar('bandit_key_type', int, str)


class Context(BaseContext):
    def __init__(
            self, bandit_selector_rule: Callable[..., bandit_key_type] = None
    ):
        BaseContext.__init__(self)
        self._bandit_selector_rule = bandit_selector_rule \
            if bandit_selector_rule is not None \
            else lambda: self.context_bandits[0]

    def take_action_with_rule(self, target: float, rule_args: Iterable[any], action_args: Iterable[any]):
        bandit_key = self._bandit_selector_rule(*rule_args)

        return self.take_action(
            bandit_key, target, *action_args
        ) if bandit_key in self.context_bandits.keys() else ex(
            exceptions.BanditKeyNotInContextException(bandit_key)
        )

    def take_action(self, bandit_key: any, target: float, *args, **kwargs):
        return self.actions[self._make_bandit_decide(bandit_key, target)].act(*args, **kwargs)

    def _make_bandit_decide(self, bandit_key: any, target: float):
        bandit_last_action_index, action_key = self.context_bandits[bandit_key].bandit.decide(
            last_action_index=self.context_bandits[bandit_key].last_action_index,
            target=target
        )
        self.context_bandits[bandit_key].last_action_index = bandit_last_action_index
        self.actions[action_key].increment_count()
        return action_key

    def set_bandit_selector(self, rule: Callable):
        self._bandit_selector_rule = rule
