from enum import Enum


class WeightSolverArgs(Enum):
    SOLVER_ID: str = "solver_id"
    ACTION_KEYS: str = "action_keys"
    OPTIMISTIC_VALUE: str = "optimistic_value"
    STEP_SIZE: str = "step_size"


class EpsilonSolverArgs(Enum):
    SOLVER_ID: str = WeightSolverArgs.SOLVER_ID.value
    ACTION_KEYS: str = WeightSolverArgs.ACTION_KEYS.value
    OPTIMISTIC_VALUE: str = WeightSolverArgs.OPTIMISTIC_VALUE.value
    STEP_SIZE: str = WeightSolverArgs.STEP_SIZE.value
    EPSILON: str = "epsilon"


class UCBSolverArgs(Enum):
    SOLVER_ID: str = WeightSolverArgs.SOLVER_ID.value
    ACTION_KEYS: str = WeightSolverArgs.ACTION_KEYS.value
    OPTIMISTIC_VALUE: str = WeightSolverArgs.OPTIMISTIC_VALUE.value
    STEP_SIZE: str = WeightSolverArgs.STEP_SIZE.value
    CONFIDENCE: str = "confidence"


class SamplingSolverArgs(Enum):
    SOLVER_ID: str = WeightSolverArgs.SOLVER_ID.value
    ACTION_KEYS: str = WeightSolverArgs.ACTION_KEYS.value
    N_SAMPLING: str = "n_sampling"
    MAX_SAMPLE_SIZE: str = "max_sample_size"
