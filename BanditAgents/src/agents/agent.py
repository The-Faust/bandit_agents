from typing import Callable, Dict, Iterable, Self, Tuple, Type

from numpy import empty, float64, int64, ndarray

from BanditAgents.src.agents.base_agent import BaseAgent
from BanditAgents.src.domain import actionKey, agentKey
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


class Agent(BaseAgent):
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
        agent_id: agentKey = False,
    ) -> None:
        """Constructor to instanciate the agent

        Parameters
        ----------
        actions : Iterable[Tuple[actionKey, Callable[[any], float]]]
            actions to be executed while the agent run
        actions_between_fits : int, optional
            Number of actions to execute before fitting the solver, by default 1
        context_hyperparameters : Type[, optional
            Hyperparameters of the context, by default ContextHyperParameters()
        solver_hyperparameters : Type[, optional
            Hyperparameters of the solver,
            can be EpsilonSolverHyperParameters,
            SamplingSolverHyperParameters,
            UCBSolverHyperParameters or
            WeightSolverHyperParameters
            by default SamplingSolverHyperParameters()
        agent_id : agentKey, optional
            Id of the agent if false it will be a UUID, by default False
        """
        super().__init__(agent_id=agent_id)

        self.actions_between_fits = actions_between_fits
        self.context = self.contexts_dict[
            type(context_hyperparameters).__name__
        ](actions=actions, **context_hyperparameters.__dict__)
        self.solver = self._from_solver_hyperparameters_make_solver(
            action_keys=self.context.get_action_keys(),
            solver_hyperparameters=solver_hyperparameters,
        )

    def act(self, *args, **kwargs) -> Tuple[ndarray[int64], ndarray[float64]]:
        """Let the solver take action on the context

        Returns
        -------
        Tuple[ndarray[int64], ndarray[float64]]
            list of action ids and associated targets
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
        """fit the solver

        Returns
        -------
        Self
            returns the newly fitted agent
        """
        self.solver.fit(*args, **kwargs)

        return self

    def info(self) -> Dict[str, any]:
        """produces information about the agent

        Returns
        -------
        Dict[str, any]
            Information of the agent as a dictionary
        """
        agent_info = dict(
            context_info=self.context.info(), solver_info=self.solver.info()
        )

        return agent_info
