import logging
from typing import Iterable

from numpy import empty, float64, int64, ndarray
from BanditAgents import Agent, actionKey
from scipy.stats import gamma

from BanditAgents.src.domain.hyperparameters import EmptyContextHyperParameters


def empty_context_agent_exemple(n_steps: int = 100):
    # Say we cannot execute the agent's choices in a reasonable delay.
    # The action execution will take more time than is allowed for the
    # timeout of your application and so the agent cannot wait all that time.
    # You will have to execute the action independently of the agent,
    # to do so you can make use of an EmptyContext object
    exemple_logger: logging.Logger = logging.getLogger(__name__)

    def action_a() -> float:
        return gamma.rvs(a=6.8, scale=0.1, loc=0, size=1)[0]

    def action_b() -> float:
        return gamma.rvs(a=2.2, scale=0.2, loc=0, size=1)[0]

    print(action_a.__name__)

    def execute_actions(action_keys: Iterable[actionKey]) -> ndarray[float64]:
        targets: ndarray[float64] = empty(len(action_keys))

        for i, action_key in enumerate(action_keys):
            if action_key == action_a.__name__:
                target: float = action_a()

            elif action_key == action_b.__name__:
                target = action_b()

            targets[i] = target

        return targets

    exemple_logger.debug(
        "Starting basic agent exemple\n"
        "---------------------------------------------------\n"
        "Initiating agent"
    )

    context_hyperparameters = EmptyContextHyperParameters(
        action_keys=('action_a', 'action_b')
    )
    agent: Agent = Agent(context_hyperparameters)

    indexes: ndarray[int64]
    targets: ndarray[float64]

    exemple_logger.debug(f"Agent initiated \n{agent.info()}")

    for i in range(n_steps):
        exemple_logger.debug(f"running step {i}")

        indexes = agent.act()
        action_keys: tuple[actionKey, ...] = (
            agent.solver.indexes_to_action_keys(indexes)
        )
        targets = execute_actions(action_keys)

        agent = agent.fit(x=indexes, y=targets)

        exemple_logger.debug(f"agent info is: {agent.info()}")

    exemple_logger.debug("---------------------------------------------------")
