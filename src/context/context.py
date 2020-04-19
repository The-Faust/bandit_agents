from typing import Callable
from random import randint
from src.context.base_context import BaseContext


# TODO: My biggest qualm with this version of context is that it is not asynchronous
#  -> This means 2 bandits cannot take action at the same time. I'll have to think about a solution

# TODO: I would LOVE to make it so that I could export contexts as pandas dataframes
#  or their equivalent in spark
class Context(BaseContext):
    # rule will be defined when creating a session
    # (should be out of context since it might depend on other inputs than rewards)
    def __init__(self, bandit_selector_rule: Callable[..., str]):
        BaseContext.__init__(self)

        # Bandit selector rule is what chooses the
        #   bandit that will take action at current epoch
        self._bandit_selector_rule: Callable = bandit_selector_rule \
            if bandit_selector_rule is not None \
            else lambda: self.context_bandits[
                randint(0, len(self.context_bandits.keys()))
            ]

    def take_action_with_rule(self, target, *args, **kwargs):
        bandit_key = self._bandit_selector_rule(*args, **kwargs)
        bandits_keys = self.context_bandits.keys()
        return self.take_action(
            bandit_key
            if bandit_key in bandits_keys
            else bandits_keys[
                randint(0, len(bandits_keys))
            ],
            target
        )

    def take_action(self, bandit_key: any, target: float):
        return self.actions[self._make_bandit_decide(bandit_key, target)].action

    # TODO: with my actual grasp on python,
    #  I might've limited myself by creating custom objects for bandits in context.
    #  I should look into other alternatives
    #  (
    #      a matrix representation seem feasible and would probably be more computation-wise effective
    #          -> then again ultra performance is not the goal of this project
    #  )
    #  Finally target will be obtained from the session after the action is taken -> target generally means reward
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
