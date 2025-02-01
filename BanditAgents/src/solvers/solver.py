from typing import Dict, Generator, Iterable, Self, Tuple

from numpy import float64, ndarray, zeros
from uuid import uuid4

from BanditAgents.src.domain import actionKey, solverKey


class Solver:
    solver_id: solverKey
    action_keys: Tuple[actionKey]
    weights: ndarray[float64]

    def __init__(
        self,
        action_keys: Iterable[actionKey],
        solver_id: solverKey = None,
        *args,
        **kwargs
    ) -> None:
        """Constructor of the Solver class

        This class is a virtual class that should b inherited by all solver classes

        Parameters
        ----------
        action_keys : Iterable[actionKey]
            keys associated to the actions the solver will operate
        """
        if solver_id is not None:
            self.solver_id = solver_id

        else:
            self.solver_id = uuid4()

        self.action_keys = tuple(ac for ac in action_keys)
        self.weights = zeros(len(self.action_keys))

    def fit(self, *args, **kwargs) -> Self:
        """Signature of the fit method, all Solvers must have a fit method implemented

        This method should be what updates the solver's weights
        given information based on its previous actions

        Returns
        -------
        Self
            returns self after training
        """

    def indexes_to_action_keys(
        self, indexes: Iterable[int]
    ) -> Tuple[actionKey, ...]:
        """convert an iterable of action indexes into a tuple of action keys

        Parameters
        ----------
        indexes : Iterable[int]
            Iterable of action indexes

        Returns
        -------
        Tuple[actionKey, ...]
            Tuple of action keys associated to every index found in indexes
        """
        action_keys = tuple(self.action_keys[index] for index in indexes)

        return action_keys

    def info(self) -> Dict[str, any]:
        """Allows access to general information about the solver

        Returns
        -------
        Dict[str, any]
            returns a dictionnary containing the information
        """
        solver_info = dict(action_keys=self.action_keys, weights=self.weights)

        return solver_info

    def predict(self) -> int:
        """Signature of the predict method, all Solvers must have a predict method implemented

        This method produces the choice of action when the solver is called

        Returns
        -------
        int
            action index of the chosen action
        """

    def _step(self, *args, **kwargs) -> Generator[bool, any, None]:
        """Signature of the _step method, all Solvers must have a _step method implemented

        This method should be executed in fit tu update the weights

        Yields
        ------
        Generator[bool, any, None]
            boolean that represents if step was completed successfully
        """
