import sys
import unittest

from Tests.messages import test_not_implemented_yet


class testBaseAgent(unittest.TestCase):
    def setUp(self) -> None:
        print(
            test_not_implemented_yet(
                self.__class__.__name__, sys._getframe().f_code.co_name
            )
        )

    def test__from_solver_hyperparameters_make_solver_succeed(self) -> None:
        print(
            test_not_implemented_yet(
                self.__class__.__name__, sys._getframe().f_code.co_name
            )
        )
