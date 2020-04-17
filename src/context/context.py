# TODO: My biggest qualm with this version of context is that it is now asynchronous
#  -> This means 2 bandits cannot take action at the same time. I'll have to think about a solution

from src.context.base_context import BaseContext


# TODO: I would LOVE to make it so that I could export contexts as pandas dataframes
#  or their equivalent in spark
class Context(BaseContext):
    def __init__(self):
        BaseContext.__init__(self)

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
