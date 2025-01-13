from enum import Enum


class WeightSolverArgs(Enum):
    ACTION_KEYS: str = "action_keys"
    OPTIMISTIC_VALUE: str = "optimistic_value"
    STEP_SIZE: str = "step_size"


class EpsilonSolverArgs(Enum):
    ACTION_KEYS: str = WeightSolverArgs.ACTION_KEYS.value
    OPTIMISTIC_VALUE: str = WeightSolverArgs.OPTIMISTIC_VALUE.value
    STEP_SIZE: str = WeightSolverArgs.STEP_SIZE.value
    EPSILON: str = "epsilon"


class UCBSolverArgs(Enum):
    ACTION_KEYS: str = WeightSolverArgs.ACTION_KEYS.value
    OPTIMISTIC_VALUE: str = WeightSolverArgs.OPTIMISTIC_VALUE.value
    STEP_SIZE: str = WeightSolverArgs.STEP_SIZE.value
    CONFIDENCE: str = "confidence"


class SamplingSolverArgs(Enum):
    ACTION_KEYS: str = WeightSolverArgs.ACTION_KEYS.value
    N_SAMPLING: str = "n_sampling"
    MAX_SAMPLE_SIZE: str = "max_sample_size"
