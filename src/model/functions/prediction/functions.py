"""
Prediction functions -> since these are very simple It was decided they could be defined as a module
"""


def step_size(
        prediction_array: [float],
        last_action_index: int,
        target: float,
        step_size_value: float
) -> [float]:
    new_prediction_array = prediction_array.copy()
    new_prediction_array[last_action_index] = _step_size(
        last_prediction=prediction_array[last_action_index],
        target=target,
        step_size=step_size_value
    )
    return new_prediction_array


def mean(
        prediction_array: [float],
        last_action_index: int,
        target: float,
        actions_count_array: [int]
):
    prediction_array[last_action_index] = _mean(
        last_prediction=prediction_array[last_action_index],
        target=target,
        step=actions_count_array[last_action_index]
    )
    return prediction_array


def _step_size(last_prediction: float, target: float, step_size: float) -> float:
    return last_prediction + step_size * (target - last_prediction)


def _mean(last_prediction: float, target: float, step: int) -> float:
    return _step_size(last_prediction, target, 1 / (step - 1))
