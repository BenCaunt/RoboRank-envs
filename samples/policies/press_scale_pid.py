from __future__ import annotations


class RobotPolicy:
    def __init__(self) -> None:
        self.integral_error = 0.0
        self.last_error = 0.0
        self.command_a = 0.0

    def step(self, motor: CurrentControlledMotor) -> None:
        target_force = motor.target_force()
        measured_force = motor.scale_force()
        error = target_force - measured_force
        self.integral_error = _clamp(self.integral_error + error * motor.dt, -0.4, 0.4)
        derivative = (error - self.last_error) / max(motor.dt, 1e-9)
        self.last_error = error

        feedforward = target_force * motor.shaft_length_m / motor.kt_nm_per_amp
        target_current = feedforward + 0.45 * error + 0.18 * self.integral_error - 0.018 * derivative
        max_slew_a = 7.5 * motor.dt
        self.command_a += _clamp(target_current - self.command_a, -max_slew_a, max_slew_a)
        motor.set_current(_clamp(self.command_a, 0.0, motor.max_current_a))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
