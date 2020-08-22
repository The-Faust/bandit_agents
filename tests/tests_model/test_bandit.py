import unittest
from numpy import argmax

from tests.mocks import mockActions
from src.model import Bandit


class TestBandit(unittest.TestCase):
    def test_decide(self):
        bandit = Bandit(
            actions_keys=list(mockActions.keys()),
            step_size=0.1,
            epsilon=0.2,
            confidence=0.7,
            optimistic_value=35,
            decision_type='greedy'
        )
        self.assertEqual(
            all(bandit._actions_speculated_rewards),
            all([35, 35, 35, 35, 35, 35, 35, 35, 35, 35])
        )

        last_action_index = 0
        reward = 0.0

        for _ in range(50000):
            last_action_index, action_key = bandit.decide(last_action_index, reward)
            reward = mockActions[action_key]()

        self.assertEqual(argmax(bandit._actions_speculated_rewards), 8)
        self.assertEqual(argmax(bandit._actions_counts), 8)
