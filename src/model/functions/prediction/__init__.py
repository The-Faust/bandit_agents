class PredictionFormulas:
    @staticmethod
    def step_size(last_prediction: float, target: float, step_size: float) -> float:
        return last_prediction + step_size * (target - last_prediction)

    @staticmethod
    def mean(last_prediction: float, target: float, step: int) -> float:
        return PredictionFormulas.step_size(
            last_prediction=last_prediction,
            target=target,
            step_size=1/(step - 1))