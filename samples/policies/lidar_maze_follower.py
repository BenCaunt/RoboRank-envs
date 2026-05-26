from __future__ import annotations

import math


class RobotPolicy:
    def __init__(self) -> None:
        self.waypoint_index = 1

    def step(self, robot) -> None:
        pose = robot.get_pose()
        target = robot.get_target()
        route = robot.route()

        if math.hypot(target.x - pose.x, target.y - pose.y) <= target.radius:
            robot.set_wheel_velocity(0.0, 0.0)
            return

        while route and self.waypoint_index < len(route) - 1:
            waypoint = route[self.waypoint_index]
            if math.hypot(waypoint.x - pose.x, waypoint.y - pose.y) >= 0.32:
                break
            self.waypoint_index += 1

        goal = route[min(self.waypoint_index, len(route) - 1)] if route else target
        dx = goal.x - pose.x
        dy = goal.y - pose.y
        distance = math.hypot(dx, dy)
        heading_error = _wrap_angle(math.atan2(dy, dx) - pose.yaw)

        scan = robot.lidar()
        front = _sector_min(scan, center_deg=0.0, half_width_deg=18.0)
        front_left = _sector_min(scan, center_deg=42.0, half_width_deg=18.0)
        front_right = _sector_min(scan, center_deg=-42.0, half_width_deg=18.0)
        left = _sector_min(scan, center_deg=90.0, half_width_deg=24.0)
        right = _sector_min(scan, center_deg=-90.0, half_width_deg=24.0)

        linear_velocity = min(0.58, 0.95 * distance)
        if abs(heading_error) > 1.15:
            linear_velocity = 0.04
        else:
            linear_velocity *= max(0.25, math.cos(heading_error))

        angular_velocity = _clamp(3.4 * heading_error, -2.5, 2.5)
        if front < 0.62:
            linear_velocity *= 0.18
            angular_velocity += 1.35 if front_left >= front_right else -1.35

        side_balance = left - right
        if min(left, right) < 1.10:
            angular_velocity += _clamp(0.42 * side_balance, -0.55, 0.55)

        left_wheel = linear_velocity - 0.5 * robot.wheel_base_m * angular_velocity
        right_wheel = linear_velocity + 0.5 * robot.wheel_base_m * angular_velocity
        robot.set_wheel_velocity(
            _clamp(left_wheel, -robot.max_wheel_velocity_mps, robot.max_wheel_velocity_mps),
            _clamp(right_wheel, -robot.max_wheel_velocity_mps, robot.max_wheel_velocity_mps),
        )


def _sector_min(scan, *, center_deg: float, half_width_deg: float) -> float:
    ray_count = len(scan)
    if ray_count == 0:
        return float("inf")
    degrees_per_ray = 360.0 / ray_count
    center_index = ray_count // 2 + round(center_deg / degrees_per_ray)
    half_width = max(1, round(half_width_deg / degrees_per_ray))
    return min(float(scan[(center_index + offset) % ray_count]) for offset in range(-half_width, half_width + 1))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
