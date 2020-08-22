import unittest

from banditMaker.model.functions.bounds.ucb import _get_upper_bound
from banditMaker.model.functions.bounds.ucb import ucb


class TestsUCB(unittest.TestCase):
    def test_ucb(self):
        prediction_array = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        action_count = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        confidence = 1.0

        test_answer = [
            10.96032279,
            9.96032279,
            8.96032279,
            7.96032279,
            6.96032279,
            5.96032279,
            4.96032279,
            3.96032279,
            2.96032279,
            1.96032279
        ]

        self.assertEqual(
            all(ucb(
                prediction_array,
                action_count,
                confidence
            )),
            all(test_answer)
        )

    def test__get_upper_bound(self):
        prediction = 10.0
        action_count = 10
        total_action_count = 10
        confidence = 1.0

        self.assertEqual(
            _get_upper_bound(
                prediction,
                action_count,
                total_action_count,
                confidence
            ),
            10.479852591218808
        )
