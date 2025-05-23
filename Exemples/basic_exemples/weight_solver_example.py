import logging
from typing import Callable, List, Tuple
from numpy import ndarray
from scipy.stats import gamma

from BanditAgents import Solvers, WeightSolver, SimulationContext, solverKey


def weight_solver_example() -> ndarray[float]:
    weight_example_logger: logging.Logger = logging.getLogger(__name__)

    # First we need some actions (any function that returns a float)
    # For our situation here we will mock the actions by using some
    # gamma distributions
    action_a: Callable[[], float] = lambda: gamma.rvs(
        a=6.8, scale=0.1, loc=0, size=1
    )[0]
    action_b: Callable[[], float] = lambda: gamma.rvs(
        a=2.2, scale=0.2, loc=0, size=1
    )[0]

    action_keys: Tuple[str, str] = ("action_a", "action_b")
    weight_example_logger.debug(action_keys)

    # We now instanciate the solver
    # as the name of the function suggest this solver is a weight solver,
    # meaning it only uses the basic weighted function to make its decisions
    weight_solver: WeightSolver = Solvers().weight_solver(
        action_keys=action_keys, optimistic_value=5.0, step_size=1e-2
    )
    weight_example_logger.debug(weight_solver)

    # This list is used to identify the actions in the context
    actions: List[Tuple[str, Callable[[], float]]] = [
        ("action_a", action_a),
        ("action_b", action_b),
    ]

    weight_example_logger.debug(actions)

    # We now instanciate the Context of the experiment
    # The context is an object that facilitates the use of the solvers
    # given a number of actions that can be called as functions.
    with SimulationContext(actions) as simulation:
        weight_example_logger.debug(simulation)

        # the simulations will run for a 100 steps.
        # which means the solver will make a 100 choices
        # and action_a or b will be called a 100 times total
        targets: Tuple[solverKey, ndarray[float]] = next(
            simulation.run(
                n_steps=100,
                solvers=[weight_solver],
                steps_by_ticks=1,
                as_dict=True,
            )
        )

        weight_example_logger.debug(targets)

    return targets[1]
