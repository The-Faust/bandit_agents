from typing import Callable, Dict, Iterable, Self, Tuple, Type

from numpy import empty, float64, int64, ndarray

from BanditAgents.src.domain import actionKey
from BanditAgents.src.domain.hyperparameters import (
    BaseSolverHyperParameters,
    ContextHyperParameters,
    EpsilonSolverHyperParameters,
    SamplingSolverHyperParameters,
    UCBSolverHyperParameters,
    WeightSolverHyperParameters,
)
from BanditAgents.src.solvers import (
    BaseSolver,
    EpsilonSolver,
    SamplingSolver,
    UCBSolver,
    WeightSolver,
)
from BanditAgents.src.contexts.context import Context


class Agent:
    actions_between_fits: int
    context: Type[(Context,)]
    solver: Type[(BaseSolver,)]
    contexts_dict: Dict[str, Callable[[any], Type[(Context,)]]] = {
        ContextHyperParameters.__name__: Context
    }
    solvers_dict: Dict[str, Callable[[any], Type[(BaseSolver,)]]] = {
        EpsilonSolverHyperParameters.__name__: EpsilonSolver,
        SamplingSolverHyperParameters.__name__: SamplingSolver,
        UCBSolverHyperParameters.__name__: UCBSolver,
        WeightSolverHyperParameters.__name__: WeightSolver,
    }

    def __init__(
        self,
        actions: Iterable[Tuple[actionKey, Callable[[any], float]]],
        actions_between_fits: int = 1,
        context_hyperparameters: Type[
            (ContextHyperParameters,)
        ] = ContextHyperParameters(),
        solver_hyperparameters: Type[
            (BaseSolverHyperParameters,)
        ] = SamplingSolverHyperParameters(),
    ) -> None:
        """_summary_

        Parameters
        ----------
        actions : Iterable[Tuple[actionKey, Callable[[any], float]]]
            _description_
        actions_between_fits : int, optional
            _description_, by default 1
        context_hyperparameters : Type[, optional
            _description_, by default ContextHyperParameters()
        solver_hyperparameters : Type[, optional
            _description_, by default SamplingSolverHyperParameters()
        """

        self.actions_between_fits = actions_between_fits
        self.context = self.contexts_dict[
            type(context_hyperparameters).__name__
        ](actions=actions, **context_hyperparameters.__dict__)
        self.solver = self.solvers_dict[type(solver_hyperparameters).__name__](
            action_keys=self.context.get_action_keys(),
            **context_hyperparameters.__dict__
        )

    def act(self, *args, **kwargs) -> Tuple[ndarray[int64], ndarray[float64]]:
        """_summary_

        Returns
        -------
        float
            _description_
        """
        action_indexes: ndarray[int64] = empty(
            self.actions_between_fits, dtype=int64
        )
        targets: ndarray[float64] = empty(self.actions_between_fits)

        for i in range(self.actions_between_fits):
            action_index = self.solver.predict()
            action_indexes[i] = action_index
            targets[i] = self.context.execute(
                action_index=action_index, *args, **kwargs
            )

        return action_indexes, targets

    def fit(self, *args, **kwargs) -> Self:
        """_summary_

        Returns
        -------
        Self
            _description_
        """
        self.solver.fit(*args, **kwargs)

        return self

    def info(self) -> Dict[str, any]:
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        agent_info = dict(
            context_info=self.context.info(), solver_info=self.solver.info()
        )

        return agent_info
