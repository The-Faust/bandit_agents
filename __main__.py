import logging

from Examples.basic_examples import run_basic_examples

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formater = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formater)

logger.addHandler(console_handler)


if __name__ == '__main__':
    examples_logger: logging.Logger = logging.getLogger(__name__)

    examples_logger.debug(run_basic_examples().head(100))
