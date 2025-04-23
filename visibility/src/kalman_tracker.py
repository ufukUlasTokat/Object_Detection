import cv2
import numpy as np

class KalmanTracker:
    def __init__(self, initial_center):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.01
        self.kalman.statePost = np.array([initial_center[0],
                                          initial_center[1], 0, 0], dtype=np.float32).reshape(4, 1)

    def predict(self):
        return self.kalman.predict()

    def correct(self, measurement):
        self.kalman.correct(measurement.reshape(2, 1).astype(np.float32))