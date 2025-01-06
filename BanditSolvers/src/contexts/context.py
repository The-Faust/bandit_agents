from typing import Callable, Iterable, Tuple

from numpy import empty, ndarray

from BanditSolvers.src.domain import actionKey


class Context:
    action_keys: ndarray[actionKey]
    actions: ndarray[Callable[[any], float]]

    def __init__(self, actions: Iterable[Tuple[actionKey, Callable[[any], float]]]) -> None:
        actions: Tuple[Tuple[actionKey, Callable[[any], float]]] = tuple(actions)

        self.action_keys = empty(len(actions), dtype='<U100')
        self.actions = empty(len(actions))

        for i, (action_key, action) in enumerate(actions):
            self.action_keys[i] = action_key
            self.actions[i] = action

    def execute(self, action_index: int, *args, **kwargs) -> float:
        target: float = self.actions[action_index](*args, **kwargs)

        return target
    
    def get_action_keys(self) -> ndarray[actionKey]:
        return self.action_keys
