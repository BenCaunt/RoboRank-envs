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

        if route:
            while self.waypoint_index < len(route) - 1:
                waypoint = route[self.waypoint_index]
                if math.hypot(waypoint.x - pose.x, waypoint.y - pose.y) >= 0.24:
                    break
                self.waypoint_index += 1
            goal = route[min(self.waypoint_index, len(route) - 1)]
        else:
            goal = target

        dx = goal.x - pose.x
        dy = goal.y - pose.y
        distance = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        heading_error = _wrap_angle(heading - pose.yaw)

        linear_velocity = min(0.65, 1.15 * distance)
        if abs(heading_error) > math.pi / 2:
            linear_velocity = 0.0
        else:
            linear_velocity *= max(0.25, math.cos(heading_error))

        scan = robot.lidar()
        if len(scan) >= 24:
            center = len(scan) // 2
            forward_ranges = scan[max(0, center - 14) : min(len(scan), center + 15)]
            if len(forward_ranges) and float(forward_ranges.min()) < 0.52:
                linear_velocity *= 0.35

        angular_velocity = _clamp(4.0 * heading_error, -2.6, 2.6)
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
