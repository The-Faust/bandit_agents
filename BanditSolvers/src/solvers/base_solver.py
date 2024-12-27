from typing import Generator, Iterable, Self, Tuple
from BanditSolvers.src.domain import actionKey


class BaseSolver:
    action_keys: Tuple[actionKey]

    def __init__(self, action_keys: Iterable[actionKey]): 
        self.action_keys = tuple(ac for ac in action_keys)
        
    def fit(self, *args, **kwargs) -> Self:
        pass

    def predict(self, *args, **kwargs) -> int:
        pass

    def _step(self, *args, **kwargs) -> Generator[bool, any, None]:
        pass