from typing import Iterable, List
import unittest
from unittest.mock import patch

from numpy import array, ndarray

from BanditAgents import actionKey

from BanditAgents.mocks.builders import BaseSolverBuilder
from BanditAgents.src.solvers.base_solver import BaseSolver
from BanditAgents.src.solvers.weight_solver import WeightSolver


BASE_SOLVER_PATH: str = f"{WeightSolver.__module__}.{BaseSolver.__name__}"


def make_mock_base_solver(action_keys: Iterable[actionKey]) -> BaseSolver:
    return BaseSolverBuilder().with_action_keys(action_keys).build()


class TestWeightSolver(unittest.TestCase):
    @patch(BASE_SOLVER_PATH, side_effect=make_mock_base_solver)
    def setUp(self, mock_base_solver) -> None:
        self.mock_action_keys: List[actionKey] = ["action_a", "action_b"]
        self.mock_optimistic_value: float = 1.0
        self.mock_step_size: float = 1.0

        self.weight_solver: WeightSolver = WeightSolver(
            action_keys=self.mock_action_keys,
            optimistic_value=self.mock_optimistic_value,
            step_size=self.mock_step_size,
        )

    def test_action_keys_to_indexes_succeed(self) -> None:
        expected_action_indexes: ndarray[int] = array([0.0, 1.0])
        action_indexes: ndarray[int] = (
            self.weight_solver.action_keys_to_indexes(self.mock_action_keys)
        )

        self.assertTrue(
            all(
                i == j for i, j in zip(action_indexes, expected_action_indexes)
            )
        )

    def test_fit_succeed(self) -> None:
        pass

    def test_predict_succeed(self) -> None:
        pass

    def test__init_weights_succeed(self) -> None:
        pass

    def test__steps_succeed(self) -> None:
        pass

    def test__step_succeed(self) -> None:
        pass

    def test__random_action_succeed(self) -> None:
        pass

    def test__compute_weight_succeed(self) -> None:
        pass
