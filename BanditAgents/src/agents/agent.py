import logging
from typing import Callable, Dict, Self, Tuple, Type

from numpy import empty, float64, int64, ndarray

from BanditAgents.src.agents.base_agent import BaseAgent
from BanditAgents.src.contexts.deployable.deployable_context import (
    DeployableContext,
)
from BanditAgents.src.contexts.deployable.empty_context import EmptyContext
from BanditAgents.src.domain import agentKey
from BanditAgents.src.domain.hyperparameters import (
    BaseSolverHyperParameters,
    DeployableContextHyperParameters,
    EmptyContextHyperParameters,
    SyncContextHyperParameters,
    SamplingSolverHyperParameters,
)
from BanditAgents.src.solvers import Solver
from BanditAgents.src.contexts.deployable.sync_context import SyncContext


class Agent(BaseAgent):
    actions_between_fits: int
    context: Type[(DeployableContext,)]
    solver: Type[(Solver,)]
    contexts_dict: Dict[str, Callable[[any], Type[(SyncContext,)]]] = {
        SyncContextHyperParameters.__name__: SyncContext,
        EmptyContextHyperParameters.__name__: EmptyContext,
    }

    def __init__(
        self,
        context_hyperparameters: Type[(DeployableContextHyperParameters,)],
        actions_between_fits: int = 1,
        solver_hyperparameters: Type[
            (BaseSolverHyperParameters,)
        ] = SamplingSolverHyperParameters(),
        agent_id: agentKey = False,
    ) -> None:
        """Constructor to instanciate the agent

        Parameters
        ----------
        context_hyperparameters : Type[(DeployableContextHyperParameters,)]
            Hyperparameters of the context
        actions_between_fits : int, optional
            Number of actions to execute before fitting the solver, by default 1
        solver_hyperparameters : Type[(BaseSolverHyperParameters,)], optional
            Hyperparameters of the solver,
            can be EpsilonSolverHyperParameters,
            SamplingSolverHyperParameters,
            UCBSolverHyperParameters or
            WeightSolverHyperParameters
            by default SamplingSolverHyperParameters()
        agent_id : agentKey, optional
            Id of the agent if false it will be a UUID, by default False
        """
        self.logger = logging.getLogger(__name__)

        super().__init__(agent_id=agent_id)

        self.actions_between_fits = actions_between_fits
        self.context = self._from_context_hyperparameters_make_context(
            context_hyperparameters=context_hyperparameters
        )
        self.solver = self._from_solver_hyperparameters_make_solver(
            action_keys=self.context.get_action_keys(),
            solver_hyperparameters=solver_hyperparameters,
        )

    def act(
        self, *args, **kwargs
    ) -> Tuple[ndarray[int64], ndarray[float64]] | ndarray[int64]:
        """Let the solver take action on the context

        Returns
        -------
        Tuple[ndarray[int64], ndarray[float64]]
            list of action ids and associated targets
        """
        is_empty_context: bool = isinstance(self.context, EmptyContext)

        results: Tuple[ndarray[int64], ndarray[float64]] | ndarray[int64]

        if is_empty_context:
            results = self.predict(*args, **kwargs)

        else:
            action_indexes: ndarray[int64] = empty(
                self.actions_between_fits, dtype=int64
            )

            targets: ndarray[float64] = empty(self.actions_between_fits)

            for i in range(self.actions_between_fits):
                action_index: int = self.solver.predict()
                action_indexes[i] = action_index
                targets[i] = self.context.execute(
                    action_index=action_index, *args, **kwargs
                )

            results = (action_indexes, targets)

        return results

    def predict(self, *args, **kwargs) -> ndarray[int64]:
        action_indexes: ndarray[int64] = empty(
            self.actions_between_fits, dtype=int64
        )

        for i in range(self.actions_between_fits):
            action_index: int = self.solver.predict()
            action_indexes[i] = action_index

        return action_indexes

    def fit(self, *args, **kwargs) -> Self:
        """Fit the solver

        Returns
        -------
        Self
            returns the newly fitted agent
        """
        self.solver.fit(*args, **kwargs)

        return self

    def info(self) -> Dict[str, any]:
        """Produces information about the agent

        Returns
        -------
        Dict[str, any]
            Information of the agent as a dictionary
        """
        agent_info = dict(
            context_info=self.context.info(), solver_info=self.solver.info()
        )

        return agent_info

    def _from_context_hyperparameters_make_context(
        self,
        context_hyperparameters: Type[(DeployableContextHyperParameters,)],
    ) -> Type[(DeployableContextHyperParameters,)]:
        """Makes a context from a context hyperparameter object

        Returns
        -------
        Type[(DeployableContextHyperParameters,)]
            An object that inherited from deployable context
        """
        context: Type[(DeployableContext,)] = self.contexts_dict[
            type(context_hyperparameters).__name__
        ](**context_hyperparameters.__dict__)

        return context
