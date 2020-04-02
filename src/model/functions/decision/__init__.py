from src.model.functions.decision.e_greedy import EGreedy
from src.model.functions.decision.ucb import UCB


class DecisionFunctions:
    def __init__(self):
        self.e_greedy = EGreedy.make_decision
        self.ucb = UCB.make_decision
