from typing import Iterable
from numpy import random
from BanditSolvers.src.solvers.weight_solver import WeightSolver
from BanditSolvers.src.domain import actionKey


class EpsilonSolver(WeightSolver):
    epsilon: float

    def __init__(
        self, 
        action_keys: Iterable[actionKey], 
        optimistic_value: float = 0., 
        step_size: float = 1., 
        epsilon: float = 1e-10
    ) -> None:
        super().__init__(
            action_keys=action_keys, 
            optimistic_value=optimistic_value, 
            step_size=step_size
        )

        self.epsilon = epsilon

    def predict(self) -> int:
        if random.random(1)[0] > self.epsilon:
            return super().predict()
        
        else:
            return random.randint(self.weights.size, size=1)[0]