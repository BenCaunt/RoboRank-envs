from __future__ import annotations

import math


class RobotPolicy:
    def step(self, robot: CollisionProbe) -> None:
        sample = robot.imu()

        planar_shock = math.hypot(sample.ax, sample.ay)
        vertical_shock = abs(sample.az - 9.81)
        gyro_shock = abs(sample.gz)
        contact = planar_shock > 8.0 or vertical_shock > 7.0 or gyro_shock > 4.0

        if planar_shock > 16.0 or vertical_shock > 10.0:
            severity = "hard"
        elif contact:
            severity = "light"
        else:
            severity = "none"

        robot.submit_collision_decision(contact, severity)
