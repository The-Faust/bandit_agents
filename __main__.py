import logging

from pandas import DataFrame, concat
from Examples.basic_examples.epsilon_solver_example import epsilon_solver_example
from Examples.basic_examples.ucb_solver_example import ucb_solver_example
from Examples.basic_examples.weight_solver_example  import weight_solver_example

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formater = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formater)

logger.addHandler(console_handler)


def run_examples() -> DataFrame:
    ws_simulation_results = DataFrame(weight_solver_example())
    ws_simulation_results['simulation'] = 'weighted'
    es_simulation_results = DataFrame(epsilon_solver_example())
    es_simulation_results['simulation'] = 'epsilon'
    ucbs_simulation_result = DataFrame(ucb_solver_example())
    ucbs_simulation_result['simulation'] = 'ucb'

    simulation_results: DataFrame = concat([
        ws_simulation_results,
        es_simulation_results,
        ucbs_simulation_result
    ])

    return simulation_results


if __name__ == '__main__':
    examples_logger: logging.Logger = logging.getLogger(__name__)

    examples_logger.debug(run_examples().head(100))
