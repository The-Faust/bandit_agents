#!/usr/bin/env python3
from tests.context_sample_run import sample_run


def main():
    sample_run(10000)


if __name__ == '__main__':
    print('    Only running a context (containing k bandits)    ')
    main()
    print('____________________________\n')