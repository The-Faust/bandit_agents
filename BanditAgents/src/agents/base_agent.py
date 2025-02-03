import logging
from typing import Callable, Dict, Tuple, Type
from uuid import uuid4

from BanditAgents.src.solvers import Solvers

from BanditAgents.src.domain import actionKey, agentKey
from BanditAgents.src.domain.hyperparameters import (
    BaseSolverHyperParameters,
    SamplingSolverHyperParameters,
    UCBSolverHyperParameters,
    WeightSolverHyperParameters,
)

from BanditAgents.src.domain.hyperparameters import (
    EpsilonSolverHyperParameters,
)
from BanditAgents.src.solvers.solver import Solver


class BaseAgent:
    agent_id: agentKey
    solvers: Solvers = Solvers()
    solvers_dict: Dict[str, Callable[[any], Type[(Solver,)]]] = {
        EpsilonSolverHyperParameters.__name__: solvers.epsilon_solver,
        SamplingSolverHyperParameters.__name__: solvers.sampling_solver,
        UCBSolverHyperParameters.__name__: solvers.ucb_solver,
        WeightSolverHyperParameters.__name__: solvers.weight_solver,
    }

    def __init__(self, agent_id: agentKey = False) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.agent_id = agent_id if agent_id else uuid4()

    def _from_solver_hyperparameters_make_solver(
        self,
        action_keys: Tuple[actionKey],
        solver_hyperparameters: Type[(BaseSolverHyperParameters,)],
    ) -> Type[(Solver)]:
        self.logger.debug(
            f"making solver from hyperparameters {solver_hyperparameters}"
        )

        solver: Type[(Solver,)] = self.solvers_dict[
            type(solver_hyperparameters).__name__
        ](action_keys=action_keys, **solver_hyperparameters.__dict__)

        return solver
