from typing import List
import unittest
from unittest.mock import patch


from BanditAgents.src.solvers.ucb_solver import UCBSolver
from BanditAgents.src.domain import actionKey
from BanditAgents.src.solvers.weight_solver import WeightSolver

WEIGHT_SOLVER_PATH: str = f'{UCBSolver.__module__}.{WeightSolver.__name__}'
WEIGHT_SOLVER__COMPUTE_WEIGHT_PATH: str = (
    f'{WEIGHT_SOLVER_PATH}.{WeightSolver._compute_weight.__name__}'
)
_COMPUTE_WEIGHT_PATH: str = (
    f'{UCBSolver.__module__}'
    f'.{UCBSolver.__name__}'
    f'.{UCBSolver._compute_weight.__name__}'
)


def mock_weights_solver__compute_weight_succeed(
    weight: float, target: float, step_size_value: float
) -> float:
    return 1


class TestUCBSolver(unittest.TestCase):
    mock_action_keys: List[actionKey]
    mock_optimistic_value: float
    mock_step_size: float
    mock_confidence: float
    ucb_solver: UCBSolver

    @patch(WEIGHT_SOLVER_PATH)
    def setUp(self, weight_solver) -> None:
        self.mock_action_keys = ['action_a', 'action_b']
        self.mock_optimistic_value = 1, 0
        self.mock_step_size = 1.0
        self.mock_confidence = 1, 0

        self.ucb_solver = UCBSolver(
            action_keys=self.mock_action_keys,
            optimistic_value=self.mock_optimistic_value,
            step_size=self.mock_step_size,
            confidence=self.mock_confidence,
        )

    @patch(_COMPUTE_WEIGHT_PATH)
    def test__step_succeed(self, _compute_weight) -> None:
        mock_target: float = 1.0
        mock_action_index: int = 0
        self.ucb_solver._compute_weight = _compute_weight

        step_taken: bool = self.ucb_solver._step(
            mock_target, mock_action_index
        )

        self.assertEqual(_compute_weight.call_count, 5)
        self.assertTrue(step_taken)

    @patch(
        WEIGHT_SOLVER__COMPUTE_WEIGHT_PATH,
        side_effect=mock_weights_solver__compute_weight_succeed,
    )
    def test__compute_weight_succeed(self, _compute_weight) -> None:
        mock_weight = 1.0
        mock_step_size_value = 1.0
        mock_confidence = 1.0
        mock_action_count = 1
        mock_total_action_count = 1
        mock_target = 1.0

        self.ucb_solver.__class__.__bases__[0]._compute_weight = (
            _compute_weight
        )

        expected_weight: float = 1.0
        weight: float = self.ucb_solver._compute_weight(
            weight=mock_weight,
            step_size_value=mock_step_size_value,
            confidence=mock_confidence,
            action_count=mock_action_count,
            total_action_count=mock_total_action_count,
            target=mock_target,
        )

        self.assertEqual(weight, expected_weight)
