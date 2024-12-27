from typing import Any, Dict, Generator, Iterable, Self, Tuple

from numpy import array, ndarray
from BanditSolvers.src.solvers.base_solver import BaseSolver
from BanditSolvers.src.domain import actionKey


class WeightSolver(BaseSolver):
    optimistic_value: float
    step_size: float
    weights: array[float]

    def __init__(
        self,
        action_keys: Iterable[actionKey],
        optimistic_value: float = 0.,
        step_size: float = 1.
    ) -> None:
        super().__init__(action_keys=action_keys)

        self.step_size = step_size
        self.optimistic_value = optimistic_value

        self._init_weights()

    def action_keys_to_indexes(self, action_keys: Iterable[actionKey]) -> array[int]:
        action_keys_index_dict: Dict[actionKey, int] = {
            action_key: i for i, action_key 
            in enumerate(self.action_keys)
        }

        action_keys_indexes: ndarray[int] = array(
            int(action_keys_index_dict[action_key]) 
            for action_key in action_keys
        )

        return action_keys_indexes

    def fit(self, x: array[int], y: array[float]) -> Self:
        training: Generator[Generator[bool, Any, None], None, None] = (
            self._step(target=target, action_index=action_index) 
            for target, action_index in zip(y, x)
        )
        training_complete = tuple(training)

        if all(training_complete):
            return self
        
        else:
            return self
        
    def indexes_to_action_keys(self, indexes: Iterable[int]) -> Tuple[actionKey, ...]:
        action_keys = tuple(self.action_keys[index] for index in indexes)

        return action_keys
        
    def predict(self) -> int:
        return self.weights.argmax()

    def _init_weights(self) -> None:
        self.weights = array(self.optimistic_value for _ in self.action_keys)
    
    def _step(self, target: float, action_index: int) -> bool:
        try:
            self.weights[action_index] = self._compute_weight(
                weight=self.weights[action_index], 
                target=target, 
                step_size_value=self.step_size
            )

            return True
        
        except Exception as e:
            return False

    @staticmethod
    def _compute_weight(weight: float, target: float, step_size_value: float) -> float:
        new_weight: float = weight + step_size_value * (target - weight)

        return new_weight
