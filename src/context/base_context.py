"""
I have a profound disdain for using types like [any] in my code I probably should create custom types for this
"""

from numpy import nditer
from src.domain import BanditPrecursor, ContextAction, ContextBandit
from src.model import Bandit
from src.Shared.exceptions.context_exceptions import ActionNotInContextException


class BaseContext(object):
    def __init__(self):
        self.actions: {any: ContextAction} = {}
        self.context_bandits: {any: ContextBandit} = {}

    def make_bandit_decide(self, bandit_key: any, target: float):
        action_index, action_key = self.context_bandits[bandit_key].bandit.decide(
            last_action_index=self.context_bandits[bandit_key].last_action_index,
            target=target
        )
        self.context_bandits[bandit_key].last_action_index = action_index
        return action_key

    def add_bandits(self, bandits_arguments: [
        [[any]], [str], [float], [float], [float], [float], [str], [str]
    ]):
        for args in nditer(bandits_arguments):
            self.add_bandit(*args)

    def add_bandit(
            self, actions_keys: [any], bandit_key: str, optimistic_value=0.0,
            step_size=0.01, epsilon=0.01, confidence=0.01,
            prediction_type='step_size', decision_type='e_greedy'
    ):
        # TODO find a more elegant way to raise exception
        self._check_if_all_actions_are_in_context(actions_keys)
        precursor = BanditPrecursor(
            actions_keys, bandit_key, optimistic_value,
            step_size, epsilon, confidence,
            prediction_type, decision_type
        )
        self.context_bandits[precursor.bandit_key] = ContextBandit(Bandit(*precursor.get_bandit_arguments()))

    def _check_if_all_actions_are_in_context(self, actions_keys: [any]):
        all(self._check_if_action_is_in_context(action_key) for action_key in actions_keys)

    def _check_if_action_is_in_context(self, action_key: any):
        if action_key in self.actions.keys(): return True
        raise ActionNotInContextException(action_key)

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
