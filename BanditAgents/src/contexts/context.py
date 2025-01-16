from inspect import signature
from typing import Callable, Dict, Iterable, Tuple

from numpy import ndarray

from BanditAgents.src.domain import actionKey


class Context:
    action_keys: Tuple[actionKey]
    actions: Tuple[Callable[[any], float]]

    def __init__(
        self, actions: Iterable[Tuple[actionKey, Callable[[any], float]]]
    ) -> None:
        """_summary_

        Parameters
        ----------
        actions : Iterable[Tuple[actionKey, Callable[[any], float]]]
            _description_
        """
        self.actions: Tuple[Callable[[any], float]] = tuple(
            action for _, action in actions
        )
        self.action_keys: Tuple[actionKey] = tuple(
            action_key for action_key, _ in actions
        )

    def execute(self, action_index: int, *args, **kwargs) -> float:
        """_summary_

        Parameters
        ----------
        action_index : int
            _description_

        Returns
        -------
        float
            _description_
        """
        target: float = self.actions[action_index](*args, **kwargs)

        return target

    def get_action_keys(self) -> ndarray[actionKey]:
        """_summary_

        Returns
        -------
        ndarray[actionKey]
            _description_
        """
        return self.action_keys

    def info(self) -> Dict[str, any]:
        context_info = dict(
            action_keys=self.action_keys,
            actions=[signature(action) for action in self.actions],
        )

        return context_info
