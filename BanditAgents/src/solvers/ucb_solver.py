from typing import Iterable

from numpy import (
    float64,
    full,
    int64,
    ndarray,
    vectorize,
    zeros,
    sqrt,
    log,
    inf,
)
from BanditAgents.src.solvers.weight_solver import WeightSolver
from BanditAgents.src.domain import actionKey, solverKey


class UCBSolver(WeightSolver):
    action_counts: ndarray[int64]
    confidence: float64
    total_action_count: int

    def __init__(
        self,
        action_keys: Iterable[actionKey],
        optimistic_value: float = 0.0,
        step_size: float = 1.0,
        confidence: float = 1.0,
        solver_id: solverKey = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        action_keys : Iterable[actionKey]
            _description_
        optimistic_value : _type_, optional
            _description_, by default 0.
        step_size : _type_, optional
            _description_, by default 1.
        confidence : float, optional
            _description_, by default 1.
        solver_id: solverKey, optional
            _description_, by default None
        """
        super().__init__(
            action_keys=action_keys,
            optimistic_value=optimistic_value,
            step_size=step_size,
            solver_id=solver_id,
        )

        self.action_counts = zeros(len(self.action_keys))
        self.confidence = confidence
        self.total_action_count = 0

    def _step(self, target: float, action_index: int) -> bool:
        """_summary_

        Parameters
        ----------
        target : float
            _description_
        action_index : int
            _description_

        Returns
        -------
        bool
            _description_
        """
        targets: ndarray[float64] = full(len(self.action_keys), -inf)
        targets[action_index] = target

        self.action_counts[action_index] += 1
        self.total_action_count += 1

        def compute_weight(
            action_weight: float, action_target: float, action_count: int
        ) -> float:
            new_weight: float = self._compute_weight(
                weight=action_weight,
                step_size_value=self.step_size,
                confidence=self.confidence,
                action_count=action_count,
                total_action_count=self.total_action_count,
                target=action_target,
            )

            return new_weight

        vec_compute_weight = vectorize(compute_weight)

        self.weights = vec_compute_weight(
            self.weights, targets, self.action_counts
        )

        return True

    @staticmethod
    def _compute_weight(
        weight: float,
        step_size_value: float,
        confidence: float,
        action_count: int,
        total_action_count: int,
        target: float = None,
    ) -> float:
        """_summary_

        Parameters
        ----------
        weight : float
            _description_
        step_size_value : float
            _description_
        confidence : float
            _description_
        action_count : int
            _description_
        total_action_count : int
            _description_
        target : float, optional
            _description_, by default None

        Returns
        -------
        float
            _description_
        """
        if target > -inf and target < inf:
            new_weight: float = WeightSolver._compute_weight(
                weight=weight, target=target, step_size_value=step_size_value
            )

        else:
            new_weight: float = weight

        upper_bound_boost_numerator: float = log(total_action_count)
        upper_bound_boost_denominator: float = (
            action_count if action_count > 0 else 1.0
        )
        upper_bound_boost: float = sqrt(
            upper_bound_boost_numerator / upper_bound_boost_denominator
        )
        new_weight = new_weight + (confidence * upper_bound_boost)

        return new_weight
