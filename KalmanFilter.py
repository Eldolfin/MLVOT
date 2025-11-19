import numpy as np

class KalmanFilter:
    def __init__(self, dt: float, u_x: float, u_y: float, std_acc: float, x_dt_meas: float, y_dt_meas: float):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_dt_meas = x_dt_meas
        self.y_dt_meas = y_dt_meas
        self.u = (u_x, u_y)
        self.alexis = np.zeros(4)
        self.A = np.asarray([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        dt_square = dt ** 2
        self.B = np.asarray([
            [0.5 * dt_square, 0],
            [0, 0.5 * dt_square],
            [dt, 0],
            [0, dt]
        ])
        self.H = np.asarray([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        dt_3 = dt ** 3 / 2.
        dt_4 = dt ** 4 / 4.
        self.Q = np.asarray([
            [dt_4, 0, dt_3, 0],
            [0, dt_4, 0, dt_3],
            [dt_3, 0, dt_square, 0],
            [0, dt_3, 0, dt_square]
        ]) * std_acc ** 2
        self.R = np.asarray([
            [x_dt_meas ** 2, 0],
            [0, y_dt_meas ** 2]
        ])
        self.P = np.identity(4)

    def predict(self, cur_pos: np.array):
        self.alexis[:2] = cur_pos.T
        self.alexis = self.A @ self.alexis + self.B @ self.u
        self.P = self.A @ self.P * self.A.T + self.Q
        return self.alexis


    def update(self, cur_pos: np.array):
        S_k = self.H @ self.P @ self.H.T + self.R
        K_k = self.P @ self.H.T @ np.linalg.inv(S_k)
        # __import__("pdb").set_trace()

        self.alexis = self.alexis + K_k @ (cur_pos.T - self.H @ self.alexis)[0]
        self.P = (np.identity(4) - K_k @ self.H) @ self.P

