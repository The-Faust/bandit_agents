# Bandit Maker #
Welcome to bandit maker!

If you don't know much about reinforcement learning I highly suggest you to read a little on the subject before using this library. 
I will do my best to keep the code retro-compatible for later releases

## When to use this library? ##
when your problem can be modeled as an agent that have to choose what action to take to maximise a given problem.

for more information see: https://en.wikipedia.org/wiki/Multi-armed_bandit (yes I cite Wikipedia and no I have no shame.)

### best used when: ###
```
1- actions to take seem to be mostly random
2- problem have a universal solution but is too difficult to express mathematically
```

## next steps (what to expect in future releases) ##
### for sure ###
```
  - refactor of how context update bandits
  - unit testing
  - documentation
  - true CI
  - add sessions to the library (object that will instantiate contexts and handle events)
```
### Not for sure (but would be neat) ###
```
  - implement true reinforcement learning (model does not just choose but plan a few steps ahead)
  - concept of regret
  - concept of drifting
  - decay (decay already exists "kindof" because the way bandit_maker learns is by according more importance to recent events that older ones)
```

## How to use the model ##
for complete implementation see the tests folder in the project.
From here ./tests/context_sample_run.py

### actions ###
Lets pretend we have some functions who return rewards said functions are
```python
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
```
our objective is to maximize such a system

### Context ###
#### init ####
we will first initialize our context object
```python
def init_context() -> Context:
    return Context()
```
#### add actions ####
we add our actions to the context
```python
def add_actions_to_context(context: Context, actions: {str: Callable[..., float]}) -> Context:
    [
        context.add_action(action_key, actions[action_key])
        for action_key
        in actions.keys()
    ]
    return context
```
### Bandits ###
#### add bandits ####
we then add the bandit to said context (here we add only one bandit to keep things simpler)
```python
def add_bandit_to_context(context: Context, bandit_key: str, actions_keys: [str]) -> Context:
    context.add_bandit(
        bandit_key=bandit_key,
        actions_keys=actions_keys,
        step_size=0.1,
        epsilon=0.2,
        confidence=0.7,
        optimistic_value=35,
        decision_type='greedy'
    )
    return context
```

### finally ###
we put it all together and run the model/train it
```python
# at a later date this function will be unnecessary because of the session object see 
#   "next steps (what to expect in future releases)" section of the README
def sample_run(n_iter):
    actions = define_actions()
    context = init_context()
    context = add_actions_to_context(context, actions)
    context = add_bandit_to_context(context, 'seller', [key for key in actions.keys()])
    print(context)

    reward = 0
    for _ in range(n_iter):
        reward = context.take_action('seller', reward)

    print(context)
```
Note: for a better definition of the code I recommend you to see the sample_run file in tests