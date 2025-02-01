from typing import Dict, Iterable, List, Tuple
from numpy import ndarray
from pandas import DataFrame, concat
from Exemples.performance_exemples.basic_performance_comparison_exemple import (
    basic_performance_comparison_exemple,
)


def run_performance_exemples() -> DataFrame:
    simulation_results: Iterable[Tuple[str, Dict[str, ndarray]]] = (
        basic_performance_comparison_exemple()
    )

    sim_dfs: List[DataFrame] = []
    for solver_id, solver_simulation_result in simulation_results:
        solver_simulation_df = DataFrame(solver_simulation_result)
        solver_simulation_df["simulation"] = solver_id

        sim_dfs = [*sim_dfs, solver_simulation_df]

    simulation_df: DataFrame = concat(sim_dfs)

    return simulation_df


__all__: List[str] = [
    "run_performance_exemples",
    "basic_performance_comparison_exemple",
]
