#!/usr/bin/env python3
import unittest

from tests.context_sample_run import sample_run

from tests.tests_model import TestsUCB
from tests.tests_model import TestsEGreedy
from tests.tests_model import TestsStepSize
from tests.tests_model import TestsMean
from tests.tests_model import TestBandit


def main():
    sample_run(10000)


if __name__ == '__main__':
    # print('    Only running a context (containing k bandits)    ')
    # main()
    unittest.main(verbosity=2)
    print('____________________________\n')