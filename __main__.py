import argparse
import logging

from Exemples.agents_exemples import run_agents_exemples
from Exemples.basic_exemples import run_basic_exemples
from Exemples.performance_exemples import run_performance_exemples

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formater = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formater)

logger.addHandler(console_handler)


parser = argparse.ArgumentParser()

parser.add_argument("-e", "--exemple")

args: argparse.Namespace = parser.parse_args()


if __name__ == "__main__":
    examples_logger: logging.Logger = logging.getLogger(__name__)

    if args.exemple == "basic":
        print(run_basic_exemples().head(100))

    elif args.exemple == "agents":
        run_agents_exemples()

    elif args.exemple == "performance":
        print(run_performance_exemples().head(100))

    else:
        print("please select an exemple to run")
