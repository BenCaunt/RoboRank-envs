import numpy as np


class RobotPolicy:
    def __init__(self):
        self.x = None
        self.P = None

    def step(self, cart):
        dt = cart.dt
        measured_position = cart.wall_position_m - cart.distance_to_wall()
        measurement_variance = cart.distance_noise_std_m**2 + (cart.distance_quantization_m**2) / 12.0

        if self.x is None:
            self.x = np.array([[measured_position], [0.0], [0.0]], dtype=float)
            self.P = np.diag([measurement_variance, 1.0, 4.0])
            cart.submit_acceleration(0.0)
            return

        F = np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        H = np.array([[1.0, 0.0, 0.0]], dtype=float)

        jerk_spectral_density = 6.0
        Q = jerk_spectral_density * np.array(
            [
                [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
                [dt**3 / 6.0, dt**2 / 2.0, dt],
            ],
            dtype=float,
        )
        R = np.array([[measurement_variance]], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        innovation = np.array([[measured_position]], dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(3) - K @ H) @ self.P

        cart.submit_acceleration(float(self.x[2, 0]))
