from numpy import Array, argmax
from random import uniform, randint


class DecisionPredictionFunctions:
    def __init__(self):
        self.decision_formulas = DecisionFormulas()
        self.prediction_formulas = PredictionFormulas()


class DecisionFormulas:
    def __init__(self):
        self.e_greedy = EGreedy.make_decision
        self.greedy = Greedy.make_decision


class Greedy:
    @staticmethod
    def make_decision(speculated_reward_array: Array[float]):
        return argmax(speculated_reward_array)


class EGreedy:
    @staticmethod
    def _is_greedy(epsilon: float) -> bool:
        return uniform(0.0, 1.0) > epsilon

    @staticmethod
    def _e_greedy_chose_action(speculated_rewards_array: Array[float], is_greedy: bool) -> int:
        return argmax(
            speculated_rewards_array
        ) if is_greedy else randint(
            0, len(speculated_rewards_array)
        )

    @staticmethod
    def make_decision(epsilon: float, speculated_reward_array: Array[float]):
        return EGreedy._e_greedy_chose_action(
            speculated_rewards_array=speculated_reward_array,
            is_greedy=EGreedy._is_greedy(epsilon=epsilon)
        )


class PredictionFormulas:
    @staticmethod
    def standard(last_prediction: float, target: float, step_size: float) -> float:
        return last_prediction + step_size * (target - last_prediction)

    @staticmethod
    def mean(last_prediction: float, target: float, step: int) -> float:
        return PredictionFormulas.standard(
            last_prediction=last_prediction,
            target=target,
            step_size=1/(step - 1))


action_decision_functions = DecisionPredictionFunctions()
