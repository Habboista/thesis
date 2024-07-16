import numpy as np

class Camera:
    def __init__(self, K: np.ndarray, R: np.ndarray, t: np.ndarray):
        self.K: np.ndarray = K
        self.R: np.ndarray = R
        self.t: np.ndarray = t