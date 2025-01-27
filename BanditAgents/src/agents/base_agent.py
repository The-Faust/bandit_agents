from typing import Callable, Dict, Tuple, Type
from uuid import uuid4

from BanditAgents.src.domain import actionKey, agentKey
from BanditAgents.src.domain.hyperparameters import (
    BaseSolverHyperParameters,
    SamplingSolverHyperParameters,
    UCBSolverHyperParameters,
    WeightSolverHyperParameters,
)
from BanditAgents.src.solvers.epsilon_solver import EpsilonSolver
from BanditAgents.src.solvers.sampling_solver import SamplingSolver
from BanditAgents.src.solvers.ucb_solver import UCBSolver
from BanditAgents.src.solvers.weight_solver import WeightSolver
from BanditAgents.src.domain.hyperparameters import (
    EpsilonSolverHyperParameters,
)
from BanditAgents.src.solvers.base_solver import BaseSolver


class BaseAgent:
    agent_id: agentKey
    solvers_dict: Dict[str, Callable[[any], Type[(BaseSolver,)]]] = {
        EpsilonSolverHyperParameters.__name__: EpsilonSolver,
        SamplingSolverHyperParameters.__name__: SamplingSolver,
        UCBSolverHyperParameters.__name__: UCBSolver,
        WeightSolverHyperParameters.__name__: WeightSolver,
    }

    def __init__(self, agent_id: agentKey = False) -> None:
        self.agent_id = agent_id if agent_id else uuid4()

    def _from_solver_hyperparameters_make_solver(
        self,
        action_keys: Tuple[actionKey],
        solver_hyperparameters: Type[(BaseSolverHyperParameters,)],
    ) -> Type[(BaseSolver)]:
        solver: Type[(BaseSolver,)] = self.solvers_dict[
            type(solver_hyperparameters).__name__
        ](action_keys=action_keys, **solver_hyperparameters.__dict__)

        return solver
