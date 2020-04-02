class PredictionFunctions:
    @staticmethod
    def step_size(
            prediction_array: [float],
            last_action_index: int,
            target: float,
            step_size: float
    ):
        prediction_array[last_action_index] = PredictionFunctions._step_size(
            last_prediction=prediction_array[last_action_index],
            target=target,
            step_size=step_size
        )
        return prediction_array

    @staticmethod
    def mean(
            prediction_array: [float],
            last_action_index: int,
            target: float,
            actions_count_array: [int]
    ):
        prediction_array[last_action_index] = PredictionFunctions._mean(
            last_prediction=prediction_array[last_action_index],
            target=target,
            step=actions_count_array[last_action_index]
        )
        return prediction_array

    @staticmethod
    def _step_size(last_prediction: float, target: float, step_size: float) -> float:
        return last_prediction + step_size * (target - last_prediction)

    @staticmethod
    def _mean(last_prediction: float, target: float, step: int) -> float:
        return PredictionFunctions._step_size(last_prediction, target, 1 / (step - 1))
