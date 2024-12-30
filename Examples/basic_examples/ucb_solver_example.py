from typing import Callable, List, Tuple
from numpy import ndarray
from scipy.stats import gamma

from BanditSolvers import Solvers, SimulationContext

def ucb_solver_example() -> ndarray[float]:
    # First we need some actions (any function that returns a float)
    # For our situation here we will mock the actions by using some gamma distributions
    action_a: Callable[[], float] = lambda: gamma.rvs(a=0.5, loc=0., scale=1., size=1)[0]
    action_b: Callable[[], float] = lambda: gamma.rvs(a=0.1, loc=0., scale=1., size=1)[0]

    action_keys: Tuple[str, str] = ('action_a', 'action_b')
    print(action_keys)

    # We now instanciate the solver
    # as the name of the function suggest this solver is a ucb solver, 
    # meaning it will use the upper bounds weighted function to make its decisions
    weight_solver = (
        Solvers().ucb_solver(
            action_keys=action_keys, 
            optimistic_value=1., 
            step_size=1e-2,
            confidence=1e-2
        )
    )
    print(weight_solver)

    # This list is used to identify the actions in the context
    actions: List[Tuple[str, Callable[[], float]]] = [
        ('action_a', action_a),
        ('action_b', action_b)
    ]
    
    print(actions)

    # We now instanciate the Context of the experiment
    # The context is an object that facilitates the use of the solvers 
    # given a number of actions that can be called as functions.
    with SimulationContext(actions) as simulation:
        print(simulation)

        # the simulations will run for a 100 steps. 
        # which means the solver will make a 100 choices 
        # and action_a or b will be called a 100 times total
        targets: ndarray[float] = simulation.run(
            n_steps=10000, 
            solver=weight_solver, 
            steps_by_ticks=1,
            as_dict=True
        )

        print(targets)

    return targets
