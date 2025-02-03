from Exemples.agents_exemples.empty_context_agent_exemple import (
    empty_context_agent_exemple,
)
from Exemples.agents_exemples.basic_agent_exemple import basic_agent_exemple


def run_agents_exemples() -> None:
    basic_agent_exemple()
    empty_context_agent_exemple()


__all__ = ['run_agents_exemples']
