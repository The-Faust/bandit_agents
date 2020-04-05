class BanditPrecursor(object):
    def __init__(
            self, actions_keys: [any], bandit_key: any, optimistic_value=0.0,
            step_size=0.01, epsilon=0.01, confidence=0.01,
            prediction_type='step_size', decision_type='e_greedy'
    ):
        self.bandit_key = bandit_key
        self.actions = actions_keys
        self.optimistic_value = optimistic_value
        self.step_size = step_size
        self.epsilon = epsilon
        self.confidence = confidence
        self.prediction_type = prediction_type
        self.decision_type = decision_type

    def get_bandit_arguments(self) -> ([any], float, float, float, float, str, str):
        return self.actions, \
               self.optimistic_value, \
               self.step_size, \
               self.epsilon, \
               self.confidence, \
               self.prediction_type, \
               self.decision_type