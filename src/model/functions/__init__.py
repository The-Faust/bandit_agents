from src.model.functions.prediction import PredictionFormulas
from src.model.functions import DecisionFormulas


class DecisionPredictionFunctions:
    def __init__(self):
        self.decision_formulas = DecisionFormulas()
        self.prediction_formulas = PredictionFormulas()


action_decision_functions = DecisionPredictionFunctions()
