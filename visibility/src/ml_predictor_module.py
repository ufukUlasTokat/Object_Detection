import numpy as np
from sklearn.linear_model import LinearRegression

class TrajectoryMLPredictor:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []
        self.model = LinearRegression()

    def compute_features(self, trajectory):
        trajectory = np.array(trajectory)
        velocity = np.diff(trajectory, axis=0, prepend=trajectory[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        return trajectory, velocity, acceleration

    def add_measurement(self, measurement):
        self.history.append(measurement)
        if len(self.history) > 100:
            self.history.pop(0)

    def train_if_possible(self):
        if len(self.history) <= self.window_size:
            return

        X = []
        y = []
        traj, vel, acc = self.compute_features(self.history)
        for i in range(len(traj) - self.window_size):
            features = []
            for j in range(self.window_size):
                idx = i + j
                features.extend(traj[idx])
                features.extend(vel[idx])
                features.extend(acc[idx])
            X.append(features)
            y.append(traj[i + self.window_size])
        self.model.fit(X, y)

    def predict_next(self):
        if len(self.history) < self.window_size:
            return None

        traj, vel, acc = self.compute_features(self.history)
        features = []
        for i in range(-self.window_size, 0):
            features.extend(traj[i])
            features.extend(vel[i])
            features.extend(acc[i])
        X_input = np.array([features])
        pred = self.model.predict(X_input)[0]
        return np.array([[pred[0]], [pred[1]], [0], [0]], dtype=np.float32)
