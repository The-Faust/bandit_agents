from enum import Enum


class WeightSolverArgs(Enum):
    ACTION_KEYS: str = 'action_keys'
    OPTIMISTIC_VALUE: str = 'optimistic_value'
    STEP_SIZE: str = 'step_size'

class EpsilonSolverArgs(Enum):
    ACTION_KEYS: str = 'action_keys'
    OPTIMISTIC_VALUE: str = 'optimistic_value'
    STEP_SIZE: str = 'step_size'
    EPSILON: str = 'epsilon'

class UCBSolverArgs(Enum):
    ACTION_KEYS: str = 'action_keys'
    OPTIMISTIC_VALUE: str = 'optimistic_value'
    STEP_SIZE: str = 'step_size'
    CONFIDENCE: str = 'confidence'