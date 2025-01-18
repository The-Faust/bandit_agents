from typing import List
import unittest
from unittest.mock import patch

from BanditAgents.src.solvers.epsilon_solver import EpsilonSolver
from BanditAgents.src.domain import actionKey
from BanditAgents.src.solvers.weight_solver import WeightSolver

WEIGHT_SOLVER_PATH: str = f'{EpsilonSolver.__module__}.{WeightSolver.__name__}'
WEIGHT_SOLVER_PREDICT_PATH: str = (
    f'{WEIGHT_SOLVER_PATH}.{WeightSolver.predict.__name__}'
)
WEIGHT_SOLVER_RANDOM_ACTION_PATH: str = (
    f'{WEIGHT_SOLVER_PATH}.{WeightSolver._random_action.__name__}'
)


def mock_predict() -> int:
    return 1


def mock_random() -> int:
    return 0


class TestEpsilonSolver(unittest.TestCase):
    epsilon_solver: EpsilonSolver
    mock_action_keys: List[actionKey]
    mock_optimistic_value: float
    mock_step_size: float
    mock_epsilon: float

    @patch(WEIGHT_SOLVER_PATH)
    def setUp(self, weight_solver) -> None:
        self.mock_action_keys = ["action_a", "action_b"]
        self.mock_optimistic_value = 1.0
        self.mock_step_size = 1.0
        self.mock_epsilon = 0.0

        self.epsilon_solver = EpsilonSolver(
            action_keys=self.mock_action_keys,
            optimistic_value=self.mock_optimistic_value,
            step_size=self.mock_step_size,
            epsilon=self.mock_epsilon,
        )

    @patch(WEIGHT_SOLVER_RANDOM_ACTION_PATH, side_effect=mock_random)
    @patch(WEIGHT_SOLVER_PREDICT_PATH, side_effect=mock_predict)
    def test_predict_succeed(self, predict, _random_action) -> None:
        self.epsilon_solver.__class__.__bases__[0].predict = predict
        self.epsilon_solver.__class__.__bases__[0]._random_action = (
            _random_action
        )

        action_index: int = self.epsilon_solver.predict()

        predict.assert_called_once()
        self.assertEqual(action_index, 1)

    @patch(WEIGHT_SOLVER_RANDOM_ACTION_PATH, side_effect=mock_random)
    @patch(WEIGHT_SOLVER_PREDICT_PATH, side_effect=mock_predict)
    def test_predict_with_random_action_succeed(self, predict, _random_action):
        self.epsilon_solver.__class__.__bases__[0].predict = predict
        self.epsilon_solver.__class__.__bases__[0]._random_action = (
            _random_action
        )
        mock_epsilon = 1.0

        self.epsilon_solver.epsilon = mock_epsilon

        action_index: int = self.epsilon_solver.predict()

        _random_action.assert_called_once()

        self.assertEqual(action_index, 0)
