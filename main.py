from src.model.bandit import Bandit


def main():
    bandit = Bandit(n_of_actions=7, optimistic_value=5)
    print(bandit.__dict__)


if __name__ == '__main__':
    main()