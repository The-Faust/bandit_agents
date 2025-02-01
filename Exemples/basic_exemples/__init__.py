from pandas import DataFrame, concat

from Exemples.basic_exemples.epsilon_solver_example import (
    epsilon_solver_example,
)
from Exemples.basic_exemples.sampling_solver_example import (
    sampling_solver_example,
)
from Exemples.basic_exemples.ucb_solver_example import ucb_solver_example
from Exemples.basic_exemples.weight_solver_example import weight_solver_example


def run_basic_exemples() -> DataFrame:
    ws_simulation_results = DataFrame(weight_solver_example())
    ws_simulation_results["simulation"] = "weighted"
    es_simulation_results = DataFrame(epsilon_solver_example())
    es_simulation_results["simulation"] = "epsilon"
    ucbs_simulation_result = DataFrame(ucb_solver_example())
    ucbs_simulation_result["simulation"] = "ucb"
    sampling_simulation_result = DataFrame(sampling_solver_example())
    sampling_simulation_result["simulation"] = "sampling"

    simulation_results: DataFrame = concat(
        [
            ws_simulation_results,
            es_simulation_results,
            ucbs_simulation_result,
            sampling_simulation_result,
        ]
    )

    return simulation_results


__all__ = ['run_basic_exemples']
