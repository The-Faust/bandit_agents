from tests import sample_run, session


def main():
    sample_run(100)


if __name__ == '__main__':
    print('    Only running a model    ')
    session()
    print('____________________________\n')

    print('    Only running a context (containing k bandits)    ')
    main()
    print('____________________________\n')
