import logging
from typing import Any, Dict, Generator, Iterable, Self

from numpy import array, ndarray, random
from BanditSolvers.src.solvers.base_solver import BaseSolver
from BanditSolvers.src.domain import actionKey


class WeightSolver(BaseSolver):
    optimistic_value: float
    step_size: float
    weights: ndarray[float]
    is_trained: bool

    def __init__(
        self,
        action_keys: Iterable[actionKey],
        optimistic_value: float = 0.,
        step_size: float = 1.
    ) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)

        super().__init__(action_keys=action_keys)

        self.step_size = step_size
        self.optimistic_value = optimistic_value
        self.is_trained = False

        self._init_weights()

    def action_keys_to_indexes(self, action_keys: Iterable[actionKey]) -> ndarray[int]:
        action_keys_index_dict: Dict[actionKey, int] = {
            action_key: i for i, action_key 
            in enumerate(self.action_keys)
        }

        action_keys_indexes: ndarray[int] = array(
            int(action_keys_index_dict[action_key]) 
            for action_key in action_keys
        )

        return action_keys_indexes

    def fit(self, x: ndarray[int], y: ndarray[float]) -> Self:
        training_complete = tuple(self._steps(x, y))

        if all(training_complete):
            if not self.is_trained:
                self.is_trained = True

            return self
        
        else:
            return self
        
    def predict(self) -> int:
        if self.is_trained:
            return self.weights.argmax()
        
        else:
            return self._random_action()

    def _init_weights(self) -> None:
        self.weights = array([self.optimistic_value for _ in self.action_keys])

    def _steps(self, x, y) -> Generator[bool, Any, None]:
        for action_index, target in zip(x, y):
            yield self._step(target=target, action_index=action_index)
    
    def _step(self, target: float, action_index: int) -> bool:
        self.weights[action_index] = self._compute_weight(
                weight=self.weights[action_index], 
                target=target, 
                step_size_value=self.step_size
            )

        return True
    
    def _random_action(self) -> int:
        return random.randint(self.weights.size, size=1)[0]

    @staticmethod
    def _compute_weight(weight: float, target: float, step_size_value: float) -> float:
        new_weight: float = weight + step_size_value * (target - weight)

        return new_weight
