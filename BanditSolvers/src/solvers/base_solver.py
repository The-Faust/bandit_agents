from typing import Generator, Iterable, Self, Tuple
from BanditSolvers.src.domain import actionKey


class BaseSolver:
    action_keys: Tuple[actionKey]

    def __init__(
            self, 
            action_keys: Iterable[actionKey], 
            *args, **kwargs
        ) -> None:
        """_summary_

        Parameters
        ----------
        action_keys : Iterable[actionKey]
            _description_
        """
        self.action_keys = tuple(ac for ac in action_keys)
        
    def fit(self, *args, **kwargs) -> Self:
        """_summary_

        Returns
        -------
        Self
            _description_
        """
        pass

    def indexes_to_action_keys(self, indexes: Iterable[int]) -> Tuple[actionKey, ...]:
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

    def predict(self) -> int:
        """_summary_

        Returns
        -------
        int
            _description_
        """
        pass

    def _step(self, *args, **kwargs) -> Generator[bool, any, None]:
        """_summary_

        Yields
        ------
        Generator[bool, any, None]
            _description_
        """
        pass