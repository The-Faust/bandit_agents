from typing import Iterable
from numpy import random
from BanditAgents.src.solvers.weight_solver import WeightSolver
from BanditAgents.src.domain import actionKey, solverKey


class EpsilonSolver(WeightSolver):
    epsilon: float

    def __init__(
        self,
        action_keys: Iterable[actionKey],
        optimistic_value: float = 0.0,
        step_size: float = 1.0,
        epsilon: float = 1e-10,
        solver_id: solverKey = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        action_keys : Iterable[actionKey]
            _description_
        optimistic_value : float, optional
            _description_, by default 0.
        step_size : float, optional
            _description_, by default 1.
        epsilon : float, optional
            _description_, by default 1e-10
        solver_id: solverKey, optional
            _description_, by default None
        """
        super().__init__(
            action_keys=action_keys,
            optimistic_value=optimistic_value,
            step_size=step_size,
            solver_id=solver_id,
        )

        self.epsilon = epsilon

    def predict(self) -> int:
        """_summary_

        Returns
        -------
        int
            _description_
        """
        if random.random(1)[0] > self.epsilon:
            return super().predict()

        else:
            return self._random_action()
