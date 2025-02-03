import logging
from typing import Callable, Dict, Iterable, Tuple, Type

from numpy import float64, int64, ndarray

from BanditAgents import BaseSolverHyperParameters, SimulationContext

from BanditAgents.src.agents.base_agent import BaseAgent
from BanditAgents.src.domain import actionKey, agentKey
from BanditAgents.src.domain.hyperparameters import SimulationParameters
from BanditAgents.src.solvers.solver import Solver


class SimulationAgent(BaseAgent):
    agent_id: agentKey
    context: SimulationContext
    solvers: list[Type[(Solver,)]]

    def __init__(
        self,
        actions: Iterable[
            Tuple[
                actionKey,
                Callable[[any], float] | Tuple[float, ...] | ndarray[float],
            ]
        ],
        solvers_hyperparameters: Iterable[Type[(BaseSolverHyperParameters)]],
        agent_id: agentKey = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        actions : Iterable[
            Tuple[
                actionKey,
                Callable[[any], float]
                |  Tuple[float, ...]
                |  ndarray[float], ]
            ]
            _description_
        solvers_hyperparameters : Iterable[Type[
            _description_
        agent_id : agentKey, optional
            _description_, by default False
        """
        self.logger: logging.Logger = logging.getLogger(__name__)

        super().__init__(agent_id=agent_id)

        self.context = SimulationContext(actions=actions)

        action_keys: Tuple[actionKey, ...] = tuple(
            action_key for action_key in self.context.action_dict.keys()
        )

        self.solvers = list(
            self._from_solver_hyperparameters_make_solver(
                action_keys=action_keys,
                solver_hyperparameters=solver_hyperparameters,
            )
            for solver_hyperparameters in solvers_hyperparameters
        )

    def run(
        self, simulation_parameters: SimulationParameters, as_dict: bool = True
    ) -> Iterable[
        Tuple[
            str,
            Dict[str, ndarray]
            | Tuple[
                ndarray[int64], ndarray[int64], ndarray[str], ndarray[float64]
            ],
        ]
    ]:
        """_summary_

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            _description_
        as_dict : bool, optional
            _description_, by default True

        Returns
        -------
        Iterable[
            Tuple[
                str,
                Dict[str, any]
                | Tuple[ndarray[int64], ndarray[int64], ndarray[str], ndarray[float64]],
            ]
        ]
            _description_
        """
        simulation_run: Iterable[
            Tuple[
                str,
                Dict[str, any]
                | Tuple[
                    ndarray[int64],
                    ndarray[int64],
                    ndarray[str],
                    ndarray[float64],
                ],
            ]
        ] = self.context.run(
            solvers=self.solvers,
            as_dict=as_dict,
            **simulation_parameters.__dict__
        )

        return simulation_run
