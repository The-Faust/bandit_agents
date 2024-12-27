from typing import Dict, Iterable
from BanditSolvers.src.domain.solver_args import EpsilonSolverArgs, WeightSolverArgs
from BanditSolvers.src.solvers.epsilon_solver import EpsilonSolver
from BanditSolvers.src.solvers.weight_solver import WeightSolver
from BanditSolvers.src.domain import actionKey


class Solvers:
    def epsilon_solver(
        self,
        action_keys: Iterable[actionKey], 
        optimistic_value: float = None, 
        step_size: float = None,
        epsilon: float = None
    ) -> EpsilonSolver:
        es_kwargs: Dict[str, Iterable[actionKey] | float] = self._make_epsilon_solver_kwargs(
            action_keys=action_keys,
            optimistic_value=optimistic_value,
            step_size=step_size,
            epsilon=epsilon
        )

        epsilon_solver = EpsilonSolver(**es_kwargs)

        return epsilon_solver

    def weight_solver(
        self,
        action_keys: Iterable[actionKey], 
        optimistic_value: float = None, 
        step_size: float = None
    ) -> WeightSolver:
        ws_kwargs: Dict[str, Iterable[actionKey] | float] = self._make_weight_solver_kwargs(
            action_keys=action_keys,
            optimistic_value=optimistic_value,
            step_size=step_size
        )

        weight_solver = WeightSolver(**ws_kwargs)

        return weight_solver

    def _make_epsilon_solver_kwargs(
        self, 
        action_keys: Iterable[actionKey], 
        optimistic_value: float = None, 
        step_size: float = None,
        epsilon: float = None
    ) -> Dict[str, Iterable[actionKey] | float]:
        es_kwargs: Dict[str, Iterable[actionKey] | float] = self._make_weight_solver_kwargs(
            action_keys=action_keys,
            optimistic_value=optimistic_value,
            step_size=step_size
        )

        if epsilon is not None:
            es_kwargs[EpsilonSolverArgs.EPSILON.value] = epsilon

        return es_kwargs

    def _make_weight_solver_kwargs(
        self,
        action_keys: Iterable[actionKey], 
        optimistic_value: float = None, 
        step_size: float = None     
    ) -> Dict[str, Iterable[actionKey] | float]:
        ws_kwargs = dict(
            action_keys=action_keys
        )

        if optimistic_value is not None:
            ws_kwargs[WeightSolverArgs.OPTIMISTIC_VALUE.value] = optimistic_value

        if step_size is not None:
            ws_kwargs[WeightSolverArgs.STEP_SIZE.value] = step_size

        return ws_kwargs

