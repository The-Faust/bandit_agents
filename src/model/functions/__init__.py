from src.model.functions.prediction import PredictionFunctions
from src.model.functions.decision import DecisionFunctions


class DecisionPredictionFunctions:
    def __init__(self):
        self.decision_formulas = DecisionFunctions()
        self.prediction_formulas = PredictionFunctions()


action_decision_functions = DecisionPredictionFunctions()
