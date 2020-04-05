from numpy import array, nditer
from src.domain import make_bandit_precursor
from src.model.bandit import Bandit


class BaseContext(object):
    def __init__(self):
        self.actions: {any: [any, int]} = {}
        self.bandits: {any: [Bandit, int]} = {}

    def make_bandit_decide(self, bandit_key: any, target: float):
        action_index, action_key = self.bandits[bandit_key][0].decide(
            last_action_index=self.bandits[bandit_key][1],
            target=target
        )
        self.bandits[bandit_key][1] = action_index
        return action_key

    def add_bandits(self, bandits_arguments: [
        [[any]], [str], [float], [float], [float], [float], [str], [str]
    ]):
        for args in nditer(bandits_arguments):
            self.add_bandit(*args)

    def add_bandit(
            self, actions: [any], bandit_key: str, optimistic_value=0.0,
            step_size=0.01, epsilon=0.01, confidence=0.01,
            prediction_type='step_size', decision_type='e_greedy'
    ):
        precursor = make_bandit_precursor(
            actions, bandit_key, optimistic_value,
            step_size, epsilon, confidence,
            prediction_type, decision_type
        )
        self.bandits[precursor.bandit_key] = array([Bandit(*precursor.get_bandit_arguments()), 0])

    def add_action(self, action_key: str, action: any):
        self.actions[action_key] = array([action, 0])

    def set_bandit_step_size(self, bandit_key: any, step_size_value: float) -> bool:
        self.bandits[bandit_key].set_step_size(step_size_value)
        return True

    def set_bandit_epsilon(self, bandit_key: any, epsilon_value: float) -> bool:
        self.bandits[bandit_key].set_epsilon(epsilon_value)
        return True

    def set_bandit_confidence(self, bandit_key: any, confidence_value) -> bool:
        self.bandits[bandit_key].set_confidence(confidence_value)
        return True
