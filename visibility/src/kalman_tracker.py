import cv2
import numpy as np
from ml_predictor_module import TrajectoryMLPredictor


class KalmanTracker:
    def __init__(self, initial_center, use_ml=True, window_size=5):
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

        self.use_ml = use_ml
        self.ml_model = TrajectoryMLPredictor(window_size)
        self.frame_count = 0

    def predict(self, skip_yolo=False):
        self.frame_count += 1

        # İlk 10 frame boyunca ML kullanma
        if self.frame_count < 10 or not self.use_ml or not skip_yolo:
            return self.kalman.predict()

        ml_prediction = self.ml_model.predict_next()
        if ml_prediction is not None:
            self.kalman.statePre = ml_prediction
            return ml_prediction
        return self.kalman.predict()

    def correct(self, measurement, used_yolo=True):
        self.kalman.correct(measurement.reshape(2, 1).astype(np.float32))
        self.ml_model.add_measurement(measurement)
        if self.use_ml and used_yolo:
            self.ml_model.train_if_possible()