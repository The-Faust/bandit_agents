from src.context import Context
from scipy.stats.distributions import gamma


class GammaGiver:
    def __init__(self, alpha, loc):
        self.alpha = alpha
        self.loc = loc

    def do(self):
        return gamma.rvs(self.alpha, self.loc)


# How to initialize a context (pretty simple)
def init_context() -> Context:
    return Context()


# how to add actions to context
def add_actions_to_context(context: Context, actions: {str: any}) -> Context:
    [
        context.add_action(action_key, actions[action_key])
        for action_key
        in actions.keys()
    ]
    return context


# how to add a bandit to context
def add_bandit_to_context(context: Context, bandit_key: str, actions_keys: [str]) -> Context:
    context.add_bandit(
        bandit_key=bandit_key,
        actions_keys=actions_keys,
        step_size=0.1,
        epsilon=0.2,
        confidence=0.7,
        optimistic_value=50,
        decision_type='greedy'
    )
    return context


# here the "actions" are gamma distributions.
# I only played with their average but other parts of their geoetry could be changed with parameters.
# For more info check out scipy stats documentation
def define_actions() -> {}:
    return {
        'sell_a': GammaGiver(4, 20).do,
        'sell_b': GammaGiver(4, 10).do,
        'sell_c': GammaGiver(4, 30).do,
        'sell_d': GammaGiver(4, 15).do,
        'sell_e': GammaGiver(4, 2).do,
        'sell_f': GammaGiver(4, 7).do,
        'sell_g': GammaGiver(4, 25).do,
        'sell_h': GammaGiver(4, 12).do,
        'sell_i': GammaGiver(4, 31).do,
        'sell_j': GammaGiver(4, 9).do
    }


def sample_run(n_iter):
    actions = define_actions()
    context = init_context()
    context = add_actions_to_context(context, actions)
    context = add_bandit_to_context(context, 'seller', [key for key in actions.keys()])
    print(context)

    reward = 0
    for _ in range(n_iter):
        reward = context.take_action('seller', reward)()

    print(context)
