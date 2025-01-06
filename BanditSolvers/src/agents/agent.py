from typing import Callable, Iterable, Self, Tuple, Type

from BanditSolvers.src.domain import actionKey
from BanditSolvers.src.solvers.base_solver import BaseSolver
from BanditSolvers.src.contexts.context import Context


class Agent:
    solver: Type[(BaseSolver,)]
    context: Type[(Context,)]

    def __init__(
        self, 
        actions: Iterable[Tuple[actionKey, Callable[[any], float]]], 
        context: Callable[[Iterable[Tuple[actionKey, Callable[[any], float]]]], Type[(Context,)]],
        solver: Callable[[Iterable[actionKey, any]], Type[(BaseSolver,)]],
        *args, **kwargs
    ) -> None:
        self.context = context(actions)
        self.solver = solver(self.context.get_action_keys(), *args, **kwargs)

    def act(self, *args, **kwargs) -> float:
        action_index: int = self.solver.predict()
        target: float = self.context.execute(action_index=action_index, *args, **kwargs)

        return target
    
    def fit(self, *args, **kwargs) -> Self:
        self.solver.fit(*args, **kwargs)

        return self
    
