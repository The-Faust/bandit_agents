from typing import Dict, Iterable, List, Tuple, Type

from numpy import ndarray
from BanditAgents import (
    BaseSolverHyperParameters,
    SamplingSolverHyperParameters,
    UCBSolverHyperParameters,
    WeightSolverHyperParameters,
)
from BanditAgents.src.agents.simulation_agent import SimulationAgent
from BanditAgents.src.domain.hyperparameters import (
    EpsilonSolverHyperParameters,
    SimulationParameters,
)


def basic_performance_comparison_exemple() -> (
    Iterable[Tuple[str, Dict[str, ndarray]]]
):
    actions: Tuple[Tuple[str, Tuple[float, float, float]]] = (
        ("action_a", (6.8, 0, 0.1)),
        ("action_b", (2.2, 0, 0.2)),
    )

    epsilon_solvere_hyperparams = EpsilonSolverHyperParameters(
        solver_id="epsilon_1"
    )
    sampling_solver_hyperparams = SamplingSolverHyperParameters(
        solver_id="sampling_1"
    )
    ucb_solver_hyperparams = UCBSolverHyperParameters(solver_id="ucb_1")
    weight_solver_hyperparams = WeightSolverHyperParameters(
        solver_id="weight_1"
    )

    solvers_hyperparams: List[Type[(BaseSolverHyperParameters,)]] = [
        epsilon_solvere_hyperparams,
        sampling_solver_hyperparams,
        ucb_solver_hyperparams,
        weight_solver_hyperparams,
    ]

    agent = SimulationAgent(
        actions=actions, solvers_hyperparameters=solvers_hyperparams
    )

    simulation_params = SimulationParameters(n_steps=1000, steps_by_ticks=5)

    simulation_results: Iterable[Tuple[str, Dict[str, ndarray]]] = agent.run(
        simulation_parameters=simulation_params
    )

    return simulation_results
