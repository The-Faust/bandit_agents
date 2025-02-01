from typing import Iterable, List
import unittest
from unittest.mock import patch

from numpy import array, ndarray

from BanditAgents import actionKey

from BanditAgents.mocks.builders import BaseSolverBuilder
from BanditAgents.src.solvers.solver import Solver
from BanditAgents.src.solvers.weight_solver import WeightSolver


BASE_SOLVER_PATH: str = f"{WeightSolver.__module__}.{Solver.__name__}"
_COMPUTE_WEIGHT_PATH: str = (
    f"{WeightSolver.__module__}"
    f".{WeightSolver.__name__}"
    f".{WeightSolver._compute_weight.__name__}"
)
_RANDOM_ACTION_PATH: str = (
    f"{WeightSolver.__module__}"
    f".{WeightSolver.__name__}"
    f".{WeightSolver._random_action.__name__}"
)
_STEP_PATH: str = (
    f"{WeightSolver.__module__}"
    f".{WeightSolver.__name__}"
    f".{WeightSolver._step.__name__}"
)
_STEPS_PATH: str = (
    f"{WeightSolver.__module__}"
    f".{WeightSolver.__name__}"
    f".{WeightSolver._steps.__name__}"
)


def make_mock_base_solver(action_keys: Iterable[actionKey]) -> Solver:
    return BaseSolverBuilder().with_action_keys(action_keys).build()


def mock__steps(x: ndarray[int], y: ndarray[float]) -> Iterable[bool]:
    return (True for _ in x)


def mock__random_action() -> int:
    return 1


class TestWeightSolver(unittest.TestCase):
    mock_action_keys: List[actionKey]
    mock_optimistic_value: float
    mock_step_size: float
    weight_solver: WeightSolver

    @patch(BASE_SOLVER_PATH, side_effect=make_mock_base_solver)
    def setUp(self, mock_base_solver) -> None:
        self.mock_action_keys = ["action_a", "action_b"]
        self.mock_optimistic_value = 1.0
        self.mock_step_size = 1.0

        self.weight_solver = WeightSolver(
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

    @patch(_STEPS_PATH, side_effect=mock__steps)
    def test_fit_succeed(self, _steps) -> None:
        self.weight_solver._steps = _steps
        mock_ids: ndarray[int] = array([0, 0, 1, 1])
        mock_targets: ndarray[float] = array([1.0, 1.0, 0.0, 0.0])

        self.weight_solver.fit(mock_ids, mock_targets)

        _steps.assert_called_with(mock_ids, mock_targets)

    @patch(_RANDOM_ACTION_PATH, side_effect=mock__random_action)
    def test_predict_succeed(self, _random_action) -> None:
        mock_weights: ndarray[float] = array([0.0, 1.0])
        self.weight_solver.weights = mock_weights
        self.weight_solver.is_trained = True

        expected_index = 1
        chosen_index: int = self.weight_solver.predict()

        self.assertEqual(chosen_index, expected_index)

    @patch(_RANDOM_ACTION_PATH, side_effect=mock__random_action)
    def test_predict_with_random_action_succeed(self, _random_action) -> None:
        self.weight_solver._random_action = _random_action

        expected_index: int = 1
        chosen_index: int = self.weight_solver.predict()

        _random_action.assert_called_once()

        self.assertEqual(chosen_index, expected_index)

    def test__init_weights_succeed(self) -> None:
        self.weight_solver._init_weights()
        initial_weights: ndarray[float] = self.weight_solver.weights
        expected_initial_weight: float = self.mock_optimistic_value

        self.assertTrue(
            all([w == expected_initial_weight for w in initial_weights])
        )

    @patch(_STEP_PATH, side_effect=lambda target, action_index: True)
    def test__steps_succeed(self, _step) -> None:
        mock_ids: ndarray[int] = array([0, 0, 1, 1])
        mock_targets: ndarray[float] = array([1.0, 1.0, 0.0, 0.0])

        self.weight_solver._step = _step
        is_success: bool = all(
            tuple(self.weight_solver._steps(mock_ids, mock_targets))
        )

        self.assertEqual(is_success, True)

    @patch(
        _COMPUTE_WEIGHT_PATH,
        side_effect=lambda weight, target, step_size_value: 1.0,
    )
    def test__step_succeed(self, _compute_weight) -> None:
        self.weight_solver._compute_weight = _compute_weight

        is_success: bool = self.weight_solver._step(1.0, 0)

        _compute_weight.assert_called_with(
            weight=self.weight_solver.weights[0],
            target=1.0,
            step_size_value=self.weight_solver.step_size,
        )

        self.assertTrue(is_success)

    def test__random_action_succeed(self) -> None:
        chosen_index = self.weight_solver._random_action()

        self.assertLessEqual(chosen_index, len(self.weight_solver.action_keys))

    def test__compute_weight_succeed(self) -> None:
        mock_weight = 1.0
        mock_target = 1.0
        mock_step_size = 1.0

        expected_weight = 1.0
        weight: float = self.weight_solver._compute_weight(
            mock_weight, mock_target, mock_step_size
        )

        self.assertEqual(weight, expected_weight)
