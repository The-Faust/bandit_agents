from typing import Dict, Generator, Iterable, Self, Tuple

from numpy import float64, ndarray, zeros
from BanditAgents.src.domain import actionKey


class BaseSolver:
    action_keys: Tuple[actionKey]
    weights: ndarray[float64]

    def __init__(
        self, action_keys: Iterable[actionKey], *args, **kwargs
    ) -> None:
        """_summary_

        Parameters
        ----------
        action_keys : Iterable[actionKey]
            _description_
        """
        self.action_keys = tuple(ac for ac in action_keys)
        self.weights = zeros(len(self.action_keys))

    def fit(self, *args, **kwargs) -> Self:
        """_summary_

        Returns
        -------
        Self
            _description_
        """

    def indexes_to_action_keys(
        self, indexes: Iterable[int]
    ) -> Tuple[actionKey, ...]:
        """_summary_

        Parameters
        ----------
        indexes : Iterable[int]
            _description_

        Returns
        -------
        Tuple[actionKey, ...]
            _description_
        """
        action_keys = tuple(self.action_keys[index] for index in indexes)

        return action_keys

    def info(self) -> Dict[str, any]:
        solver_info = dict(action_keys=self.action_keys, weights=self.weights)

        return solver_info

    def predict(self) -> int:
        """_summary_

        Returns
        -------
        int
            _description_
        """

    def _step(self, *args, **kwargs) -> Generator[bool, any, None]:
        """_summary_

        Yields
        ------
        Generator[bool, any, None]
            _description_
        """
