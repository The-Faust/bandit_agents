from typing import Callable, List

from banditMaker.domain import BanditPrecursor, ContextAction, ContextBandit
from banditMaker.model.bandit import Bandit
from banditMaker.shared import ex
from banditMaker.shared.exceptions import context_exceptions as exceptions
from banditMaker.domain.types import action_key_type, bandit_key_type


class BaseContext(object):
    def __init__(self):
        self.actions: {action_key_type: ContextAction} = {}
        self.context_bandits: {bandit_key_type: ContextBandit} = {}

    def add_bandit(
            self, actions_keys: List[action_key_type],
            bandit_key: bandit_key_type,
            optimistic_value: float = 0.0,
            step_size: float = 0.01,
            epsilon: float = 0.01,
            confidence: float = 0.01,
            bound_type: str = 'ucb',
            prediction_type: str = 'step_size',
            decision_type: str = 'e_greedy'
    ):
        self._check_if_all_actions_are_in_context(actions_keys)
        map(
            lambda action_key: self.actions[action_key].add_bandit_key(bandit_key),
            actions_keys
        )

        precursor = BanditPrecursor(
            actions_keys,
            bandit_key,
            optimistic_value,
            step_size,
            epsilon,
            confidence,
            bound_type,
            prediction_type,
            decision_type
        )
        self.context_bandits[precursor.bandit_key] = ContextBandit(Bandit(*precursor.get_bandit_arguments()))

    def _check_if_all_actions_are_in_context(self, actions_keys: [any]):
        all(self._check_if_action_is_in_context(action_key)for action_key in actions_keys)

    def _check_if_action_is_in_context(self, action_key: any):
        return True if action_key in self.actions.keys() else ex(
            exceptions.ActionNotInContextException(action_key)
        )

    def add_action(self, action_key: str, action: Callable[..., float]):
        self.actions[action_key] = ContextAction(action_key, action)

    def set_bandit_step_size(self, bandit_key: any, step_size_value: float) -> bool:
        self.context_bandits[bandit_key].set_step_size(step_size_value)
        return True

    def set_bandit_epsilon(self, bandit_key: any, epsilon_value: float) -> bool:
        self.context_bandits[bandit_key].set_epsilon(epsilon_value)
        return True

    def set_bandit_confidence(self, bandit_key: any, confidence_value) -> bool:
        self.context_bandits[bandit_key].set_confidence(confidence_value)
        return True

    def __str__(self):
        bandits = '\n\n'.join([
            str(bandit)
            for bandit
            in self.context_bandits.values()
        ])

        return 'Context: \n' \
               '    with actions: {} \n\n' \
               '    with bandits: {} \n\n' \
            .format(self.actions, bandits)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
