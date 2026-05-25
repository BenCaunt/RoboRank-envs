from __future__ import annotations

import math


class RobotPolicy:
    def step(self, robot: DifferentialDrive) -> None:
        pose = robot.get_pose()
        target = robot.get_target()

        dx = target.x - pose.x
        dy = target.y - pose.y
        distance = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(heading - pose.yaw)

        if distance <= target.radius:
            robot.set_wheel_velocity(0.0, 0.0)
            return

        linear_velocity = min(0.95, 1.25 * distance)
        if abs(heading_error) > math.pi / 2:
            linear_velocity = 0.0
        else:
            linear_velocity *= max(0.2, math.cos(heading_error))

        angular_velocity = max(-3.0, min(3.0, 4.5 * heading_error))
        left = linear_velocity - 0.5 * robot.wheel_base_m * angular_velocity
        right = linear_velocity + 0.5 * robot.wheel_base_m * angular_velocity
        robot.set_wheel_velocity(
            _clamp(left, -robot.max_wheel_velocity_mps, robot.max_wheel_velocity_mps),
            _clamp(right, -robot.max_wheel_velocity_mps, robot.max_wheel_velocity_mps),
        )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
