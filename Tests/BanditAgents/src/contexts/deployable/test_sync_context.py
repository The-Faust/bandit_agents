import sys
import unittest

from Tests.messages import test_not_implemented_yet


class TestSyncContext(unittest.TestCase):
    def setUp(self):
        print(
            test_not_implemented_yet(
                self.__class__.__name__, sys._getframe().f_code.co_name
            )
        )

    def test_execute_succeed(self):
        print(
            test_not_implemented_yet(
                self.__class__.__name__, sys._getframe().f_code.co_name
            )
        )

    def test_info_succeed(self):
        print(
            test_not_implemented_yet(
                self.__class__.__name__, sys._getframe().f_code.co_name
            )
        )
