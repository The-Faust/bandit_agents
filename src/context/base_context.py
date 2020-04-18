"""
I have a profound disdain for using types like [any] in my code.
I probably should create custom types for this
"""

from src.domain import BanditPrecursor, ContextAction, ContextBandit
from src.model import Bandit
from src.Shared.exceptions.context_exceptions import ActionNotInContextException


class BaseContext(object):
    def __init__(self):
        self.actions: {any: ContextAction} = {}
        self.context_bandits: {any: ContextBandit} = {}

    def add_bandit(
            self, actions_keys: [any],
            bandit_key: str,
            optimistic_value: float = 0.0,
            step_size: float = 0.01,
            epsilon: float = 0.01,
            confidence: float = 0.01,
            bound_type: str = 'ucb',
            prediction_type: str = 'step_size',
            decision_type: str = 'e_greedy'
    ):
        # TODO: find a more elegant way to raise exception
        self._check_if_all_actions_are_in_context(actions_keys)

        # TODO: existence of banditPrecursor might be a bit of over-engineering I possibly will have to remove it.
        #  It is a remnant of when I wanted to define bandits as a matrix that would've been fed to the context object
        #  which would then assemble all of them at once.
        precursor = BanditPrecursor(
            actions_keys, bandit_key, optimistic_value,
            step_size, epsilon, confidence,
            bound_type, prediction_type, decision_type
        )
        self.context_bandits[precursor.bandit_key] = ContextBandit(Bandit(*precursor.get_bandit_arguments()))

    def _check_if_all_actions_are_in_context(self, actions_keys: [any]):
        all(self._check_if_action_is_in_context(action_key) for action_key in actions_keys)

    # TODO: I tend to dislike if statements and want to limit them in my code.
    #  If I find a better alternative for this I'll change this method
    def _check_if_action_is_in_context(self, action_key: any):
        if action_key in self.actions.keys():
            return True
        raise ActionNotInContextException(action_key)

    # TODO: I would recommend actual actions to be functions,
    #  but I don't want to be restrictive in usage of the library
    def add_action(self, action_key: str, action: any):
        self.actions[action_key] = ContextAction(action)

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

    # TODO: Although not the best __eq__ function it still better than nothing
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
