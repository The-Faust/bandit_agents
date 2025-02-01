from typing import Dict, Iterable
from BanditAgents.src.domain.solver_args import (
    EpsilonSolverArgs,
    SamplingSolverArgs,
    UCBSolverArgs,
    WeightSolverArgs,
)
from BanditAgents.src.solvers.solver import Solver
from BanditAgents.src.solvers.epsilon_solver import EpsilonSolver
from BanditAgents.src.solvers.sampling_solver import SamplingSolver
from BanditAgents.src.solvers.ucb_solver import UCBSolver
from BanditAgents.src.solvers.weight_solver import WeightSolver
from BanditAgents.src.domain import actionKey, solverKey


class Solvers:
    def epsilon_solver(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        optimistic_value: float = None,
        step_size: float = None,
        epsilon: float = None,
    ) -> EpsilonSolver:
        es_kwargs: Dict[str, Iterable[actionKey] | float] = (
            self._make_epsilon_solver_kwargs(
                action_keys=action_keys,
                solver_id=solver_id,
                optimistic_value=optimistic_value,
                step_size=step_size,
                epsilon=epsilon,
            )
        )

        epsilon_solver = EpsilonSolver(**es_kwargs)

        return epsilon_solver

    def sampling_solver(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        n_sampling: int = None,
        max_sample_size: int = None,
    ) -> SamplingSolver:
        sampling_kwargs: Dict[str, Iterable[actionKey] | float | int] = (
            self._make_sampling_solver_kwargs(
                action_keys=action_keys,
                solver_id=solver_id,
                n_sampling=n_sampling,
                max_sample_size=max_sample_size,
            )
        )

        sampling_solver = SamplingSolver(**sampling_kwargs)

        return sampling_solver

    def ucb_solver(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        optimistic_value: float = None,
        step_size: float = None,
        confidence: float = None,
    ) -> UCBSolver:
        ucb_kwargs: Dict[str, Iterable[actionKey] | float] = (
            self._make_ucb_solver_kwargs(
                action_keys=action_keys,
                solver_id=solver_id,
                optimistic_value=optimistic_value,
                step_size=step_size,
                confidence=confidence,
            )
        )

        ucb_solver = UCBSolver(**ucb_kwargs)

        return ucb_solver

    def weight_solver(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        optimistic_value: float = None,
        step_size: float = None,
    ) -> WeightSolver:
        ws_kwargs: Dict[str, Iterable[actionKey] | float] = (
            self._make_weight_solver_kwargs(
                action_keys=action_keys,
                solver_id=solver_id,
                optimistic_value=optimistic_value,
                step_size=step_size,
            )
        )

        weight_solver = WeightSolver(**ws_kwargs)

        return weight_solver

    def _make_epsilon_solver_kwargs(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        optimistic_value: float = None,
        step_size: float = None,
        epsilon: float = None,
    ) -> Dict[str, Iterable[actionKey] | float]:
        es_kwargs: Dict[str, Iterable[actionKey] | float] = (
            self._make_weight_solver_kwargs(
                action_keys=action_keys,
                solver_id=solver_id,
                optimistic_value=optimistic_value,
                step_size=step_size,
            )
        )

        if epsilon is not None:
            es_kwargs[EpsilonSolverArgs.EPSILON.value] = epsilon

        return es_kwargs

    def _make_sampling_solver_kwargs(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        n_sampling: int = None,
        max_sample_size: int = None,
    ) -> Dict[str, Iterable[actionKey] | float]:
        sampling_kwargs = dict(action_keys=action_keys)

        if solver_id is not None:
            sampling_kwargs[WeightSolverArgs.SOLVER_ID.value] = solver_id

        if n_sampling is not None:
            sampling_kwargs[SamplingSolverArgs.N_SAMPLING.value] = n_sampling

        if max_sample_size is not None:
            sampling_kwargs[SamplingSolverArgs.MAX_SAMPLE_SIZE.value] = (
                max_sample_size
            )

        return sampling_kwargs

    def _make_ucb_solver_kwargs(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        optimistic_value: float = None,
        step_size: float = None,
        confidence: float = None,
    ) -> Dict[str, Iterable[actionKey] | float]:
        ucb_kwargs: Dict[str, Iterable[actionKey] | float] = (
            self._make_weight_solver_kwargs(
                action_keys=action_keys,
                solver_id=solver_id,
                optimistic_value=optimistic_value,
                step_size=step_size,
            )
        )

        if confidence is not None:
            ucb_kwargs[UCBSolverArgs.CONFIDENCE.value] = confidence

        return ucb_kwargs

    def _make_weight_solver_kwargs(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        optimistic_value: float = None,
        step_size: float = None,
    ) -> Dict[str, Iterable[actionKey] | float]:
        ws_kwargs = dict(action_keys=action_keys)

        if solver_id is not None:
            ws_kwargs[WeightSolverArgs.SOLVER_ID.value] = solver_id

        if optimistic_value is not None:
            ws_kwargs[WeightSolverArgs.OPTIMISTIC_VALUE.value] = (
                optimistic_value
            )

        if step_size is not None:
            ws_kwargs[WeightSolverArgs.STEP_SIZE.value] = step_size

        return ws_kwargs


__all__: list[str] = [
    "Solver",
    "EpsilonSolver",
    "SamplingSolver",
    "UCBSolver",
    "WeightSolver",
    "Solvers",
]
