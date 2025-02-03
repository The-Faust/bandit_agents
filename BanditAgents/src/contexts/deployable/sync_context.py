from inspect import signature
from typing import Callable, Dict, Iterable, Tuple

from BanditAgents.src.contexts.deployable.deployable_context import (
    DeployableContext,
)
from BanditAgents.src.domain import actionKey


class SyncContext(DeployableContext):
    actions: Tuple[Callable[[any], float]]

    def __init__(
        self, actions: Iterable[Tuple[actionKey, Callable[[any], float]]]
    ) -> None:
        """Constructor of the SyncContext class

        The sync context is a context where the result of an action can
        be expected to be obtained instantaneously or alomost instantaneously

        Parameters
        ----------
        actions : Iterable[Tuple[actionKey, Callable[[any], float]]]
            An action consist of a pair with an actionKey at index 0
            and a callable that returns a float at index 1

            The return of the Callable will be used as
            a prediction/choice's target
        """
        self.actions: Tuple[Callable[[any], float]] = tuple(
            action for _, action in actions
        )

        super().__init__(action_key for action_key, _ in actions)

    def execute(self, action_index: int, *args, **kwargs) -> float:
        """Execute an action.

        Which meant execute actions[action_index]

        Parameters
        ----------
        action_index : int
            position of the action in the context

        Returns
        -------
        float
            target associated to the action's execution
        """
        target: float = self.actions[action_index](*args, **kwargs)

        return target

    def info(self) -> Dict[str, any]:
        context_info: Dict[str, any] = super().info()
        context_info['actions'] = [
            signature(action) for action in self.actions
        ]

        return context_info
