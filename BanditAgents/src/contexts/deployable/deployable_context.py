from typing import Iterable
from BanditAgents.src.domain import actionKey
from BanditAgents.src.contexts.context import Context


class DeployableContext(Context):
    action_keys: tuple[actionKey, ...]

    def __init__(self, action_keys: Iterable[actionKey]) -> None:
        self.action_keys = tuple(action_key for action_key in action_keys)

    def get_action_keys(self) -> tuple[actionKey, ...]:
        """Gets action keys

        Returns
        -------
        ndarray[actionKey]
            an array of action keys
        """
        return self.action_keys

    def info(self) -> dict[str, tuple[actionKey, ...]]:
        return dict(action_keys=self.action_keys)
