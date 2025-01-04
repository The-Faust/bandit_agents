import unittest

from BanditSolvers.src.solvers.base_solver import BaseSolver


class TestBaseSolver(unittest.TestCase):
    def setUp(self):
        self.base_solver = BaseSolver()

    def test_indexes_to_action_keys_succeed(self):
        