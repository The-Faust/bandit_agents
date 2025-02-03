import logging
from typing import Callable, Tuple
from numpy import float64, int64, ndarray
from scipy.stats import gamma

from BanditAgents import Agent, actionKey, SyncContextHyperParameters


def basic_agent_exemple(
    actions: Tuple[Tuple[actionKey, Callable[[], float]]] = (
        ("action_a", lambda: gamma.rvs(a=6.8, scale=0.1, loc=0, size=1)[0]),
        ("action_b", lambda: gamma.rvs(a=2.2, scale=0.2, loc=0, size=1)[0]),
    ),
    n_steps: int = 100,
) -> None:
    exemple_logger: logging.Logger = logging.getLogger(__name__)

    exemple_logger.debug(
        "Starting basic agent exemple\n"
        "---------------------------------------------------\n"
        "Initiating agent"
    )

    # Now we make the agent, the default context is of type SyncContext,
    # the default solver is a SamplingSolver
    sync_context_hyperparameters = SyncContextHyperParameters(actions)
    agent: Agent = Agent(sync_context_hyperparameters)

    indexes: ndarray[int64]
    targets: ndarray[float64]

    exemple_logger.debug(f"Agent initiated \n{agent.info()}")

    for i in range(n_steps):
        exemple_logger.debug(f"running step {i}")
        indexes, targets = agent.act()

        agent = agent.fit(x=indexes, y=targets)

        exemple_logger.debug(f"agent info is: {agent.info()}")

    exemple_logger.debug("---------------------------------------------------")
