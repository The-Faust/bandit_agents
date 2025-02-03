from typing import Iterable
from BanditAgents.src.contexts.deployable.deployable_context import (
    DeployableContext,
)
from BanditAgents.src.domain import actionKey


class EmptyContext(DeployableContext):
    def __init__(self, action_keys: Iterable[actionKey]) -> None:
        """_summary_

        Parameters
        ----------
        action_keys : Iterable[actionKey]
            keys associated to the actions that will be called
        """
        super().__init__(action_keys=action_keys)
