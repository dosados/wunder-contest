import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Initialize your model, load weights, etc.
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # This is where your prediction logic goes.
        if not data_point.need_prediction:
            return None

        # When a prediction is needed, return a numpy array of length 2 (for t0 and t1).
        # Replace this with your model's actual output.
        prediction = np.zeros(2)
        return prediction