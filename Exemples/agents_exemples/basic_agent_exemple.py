import logging
from typing import Callable, Tuple
from scipy.stats import gamma

from BanditAgents import Agent
from BanditAgents.src.domain import actionKey


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
    # Now we make the agent, the default context is of type Context,
    # the default solver is a SamplingSolver
    agent = Agent(actions)
    exemple_logger.debug(f"Agent initiated \n{agent.info()}")

    for i in range(n_steps):
        exemple_logger.debug(f"running step {i}")
        indexes, targets = agent.act()
        agent = agent.fit(x=indexes, y=targets)

        exemple_logger.debug(f"agent info is: {agent.info()}")

    exemple_logger.debug("---------------------------------------------------")
