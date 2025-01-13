from dataclasses import dataclass


@dataclass
class BaseSolverHyperParameters:
    pass


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
