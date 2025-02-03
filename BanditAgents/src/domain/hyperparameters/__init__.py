from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

from BanditAgents.src.domain import actionKey, solverKey


@dataclass
class BaseSolverHyperParameters:
    solver_id: solverKey = False


@dataclass
class SamplingSolverHyperParameters(BaseSolverHyperParameters):
    n_sampling: int = None
    max_sample_size: int = None


@dataclass
class WeightSolverHyperParameters(BaseSolverHyperParameters):
    optimistic_value: float = None
    step_size: float = None


@dataclass
class EpsilonSolverHyperParameters(WeightSolverHyperParameters):
    epsilon: float = None


@dataclass
class UCBSolverHyperParameters(WeightSolverHyperParameters):
    confidence: float = None


@dataclass
class DeployableContextHyperParameters:
    pass


@dataclass
class SyncContextHyperParameters(DeployableContextHyperParameters):
    actions: Iterable[Tuple[actionKey, Callable[[any], float]]]


@dataclass
class EmptyContextHyperParameters(DeployableContextHyperParameters):
    action_keys: Iterable[actionKey]


@dataclass
class SimulationParameters:
    n_steps: int
    steps_by_ticks: int


__all__: list[str] = [
    "BaseSolverHyperParameters",
    "SamplingSolverHyperParameters",
    "WeightSolverHyperParameters",
    "EpsilonSolverHyperParameters",
    "UCBSolverHyperParameters",
    "SyncContextHyperParameters",
    "SimulationParameters",
    "DeployableContextHyperParameters",
    "SyncContextHyperParameters",
    "EmptyContextHyperParameters",
]
