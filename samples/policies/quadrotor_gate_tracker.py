from __future__ import annotations

import math


class RobotPolicy:
    def step(self, robot: Quadrotor) -> None:
        pose = robot.get_pose()
        gate = robot.get_next_gate()

        normal_x = math.cos(gate.yaw)
        normal_y = math.sin(gate.yaw)
        target_x = gate.x + 0.34 * normal_x
        target_y = gate.y + 0.34 * normal_y
        target_z = gate.z

        dx = target_x - pose.x
        dy = target_y - pose.y
        dz = target_z - pose.z

        desired_vx = _clamp(1.05 * dx, -0.25, 0.78)
        desired_vy = _clamp(1.05 * dy, -0.42, 0.42)
        ax_cmd = _clamp(1.7 * (desired_vx - pose.vx) + 0.25 * dx, -2.2, 2.2)
        ay_cmd = _clamp(1.7 * (desired_vy - pose.vy) + 0.25 * dy, -2.0, 2.0)

        cy = math.cos(pose.yaw)
        sy = math.sin(pose.yaw)
        target_pitch = _clamp((cy * ax_cmd + sy * ay_cmd) / 9.81, -0.24, 0.24)
        target_roll = _clamp((sy * ax_cmd - cy * ay_cmd) / 9.81, -0.24, 0.24)
        yaw_error = _wrap_angle(gate.yaw - pose.yaw)

        roll_rate = _clamp(4.2 * (target_roll - pose.roll), -robot.max_body_rate_radps, robot.max_body_rate_radps)
        pitch_rate = _clamp(4.2 * (target_pitch - pose.pitch), -robot.max_body_rate_radps, robot.max_body_rate_radps)
        yaw_rate = _clamp(1.8 * yaw_error, -1.4, 1.4)

        az_cmd = _clamp(3.0 * dz - 1.15 * pose.vz, -3.0, 3.0)
        tilt_compensation = max(0.72, math.cos(pose.roll) * math.cos(pose.pitch))
        power = robot.hover_power / tilt_compensation + az_cmd / (2.0 * 9.81)
        robot.set_body_rate_and_power(
            roll_rate,
            pitch_rate,
            yaw_rate,
            _clamp(power, 0.16, robot.max_power),
        )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
