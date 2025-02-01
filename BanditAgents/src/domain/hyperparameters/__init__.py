from dataclasses import dataclass

from BanditAgents.src.domain import solverKey


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
class ContextHyperParameters:
    pass


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
    "ContextHyperParameters",
    "SimulationParameters",
]
