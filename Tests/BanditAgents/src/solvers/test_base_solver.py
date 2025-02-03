from typing import Dict, List, Tuple
import unittest

from numpy import array, float64, ndarray

from BanditAgents import Solver, actionKey


class TestBaseSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_action_keys: List[actionKey] = ["action_a", "action_b"]
        self.base_solver: Solver = Solver(self.mock_action_keys)

    def test_indexes_to_action_keys_succeed(self) -> None:
        expected_action_keys: Tuple[actionKey, ...] = tuple(
            self.mock_action_keys
        )
        action_keys: Tuple[actionKey, ...] = (
            self.base_solver.indexes_to_action_keys([0, 1])
        )

        self.assertEqual(action_keys, expected_action_keys)

    def test_info_succeed(self) -> None:
        mock_weights: ndarray[float64] = array([0.0, 0.0])
        expected_info: Dict[str, any] = {
            "action_keys": tuple(self.mock_action_keys),
            "weights": mock_weights,
        }

        info: Dict[str, any] = self.base_solver.info()

        self.assertTrue(info["action_keys"] == expected_info["action_keys"])
        self.assertTrue(
            all(
                i == j
                for i, j in zip(info["weights"], expected_info["weights"])
            )
        )
