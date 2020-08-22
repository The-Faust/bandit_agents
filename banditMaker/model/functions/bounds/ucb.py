from math import log, sqrt
from numpy import nditer, sum, array


def _get_upper_bound(
        prediction: float,
        action_count: int,
        total_actions_count: int,
        confidence: float
):
    return prediction + confidence * (
        sqrt(log(total_actions_count) / action_count)
        if action_count > 0
        else 1
    )


def ucb(
        prediction_array: [float],
        action_counts: [int],
        confidence: float,
):
    total_actions_count = int(sum(action_counts))
    return array([
        _get_upper_bound(prediction, action_count, total_actions_count, confidence)
        for prediction, action_count
        in nditer([prediction_array, action_counts])
    ])
