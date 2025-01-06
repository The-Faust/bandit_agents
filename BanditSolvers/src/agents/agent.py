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
        solver: Callable[[Iterable[actionKey], any], Type[(BaseSolver,)]],
        *args, **kwargs
    ) -> None:
        """_summary_

        Parameters
        ----------
        actions : Iterable[Tuple[actionKey, Callable[[any], float]]]
            _description_
        context : Callable[[Iterable[Tuple[actionKey, Callable[[any], float]]]], Type[
            _description_
        solver : Callable[[Iterable[actionKey], any], Type[
            _description_
        """
        self.context = context(actions)
        self.solver = solver(self.context.get_action_keys(), *args, **kwargs)

    def act(self, *args, **kwargs) -> float:
        """_summary_

        Returns
        -------
        float
            _description_
        """
        action_index: int = self.solver.predict()
        target: float = self.context.execute(action_index=action_index, *args, **kwargs)

        return target
    
    def fit(self, *args, **kwargs) -> Self:
        """_summary_

        Returns
        -------
        Self
            _description_
        """
        self.solver.fit(*args, **kwargs)

        return self
    
