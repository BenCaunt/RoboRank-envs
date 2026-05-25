from __future__ import annotations

import math


class RobotPolicy:
    def step(self, robot: DifferentialDrive) -> None:
        estimate = robot.charger_pose()
        if estimate is None:
            robot.set_wheel_velocity(-0.08, 0.08)
            return

        distance = math.hypot(estimate.x, estimate.y)
        bearing = math.atan2(estimate.y, max(0.05, estimate.x))

        if distance < 0.052 and abs(estimate.yaw) < math.radians(2.5):
            robot.set_wheel_velocity(0.0, 0.0)
            return

        linear_velocity = max(-0.10, min(0.42, 0.68 * estimate.x))
        if abs(bearing) > 0.75:
            linear_velocity = 0.0
        if distance < 0.32:
            linear_velocity = min(linear_velocity, 0.16)

        angular_velocity = _clamp(2.4 * bearing - 1.25 * estimate.yaw, -2.4, 2.4)
        left = linear_velocity - 0.5 * robot.wheel_base_m * angular_velocity
        right = linear_velocity + 0.5 * robot.wheel_base_m * angular_velocity
        robot.set_wheel_velocity(
            _clamp(left, -robot.max_wheel_velocity_mps, robot.max_wheel_velocity_mps),
            _clamp(right, -robot.max_wheel_velocity_mps, robot.max_wheel_velocity_mps),
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
