from typing import Callable
from banditMaker.model.functions.decision.e_greedy import e_greedy

decision_functions_dict: {str: Callable[..., int]} = {
    'e_greedy': e_greedy,
    'greedy': e_greedy
}
