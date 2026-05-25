import math


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class RobotPolicy:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_left_ticks = None
        self.last_right_ticks = None
        self.history = []

    def step(self, robot):
        self._predict_from_drivetrain(robot)
        self._record_history(robot.time)

        for measurement in robot.april_tag_measurements():
            self._fuse_april_tag(measurement)

        robot.submit_odometry(Pose2d(self.x, self.y, self.yaw))

    def _predict_from_drivetrain(self, robot):
        left_ticks, right_ticks = robot.get_encoder_values()
        if self.last_left_ticks is None:
            self.last_left_ticks = left_ticks
            self.last_right_ticks = right_ticks
            return

        delta_left_ticks = left_ticks - self.last_left_ticks
        delta_right_ticks = right_ticks - self.last_right_ticks
        self.last_left_ticks = left_ticks
        self.last_right_ticks = right_ticks

        meters_per_tick = (2.0 * math.pi * robot.wheel_radius_m) / robot.ticks_per_rev
        left_distance = delta_left_ticks * meters_per_tick
        right_distance = delta_right_ticks * meters_per_tick
        distance = 0.5 * (left_distance + right_distance)
        yaw_delta = robot.gyro() * robot.dt
        mid_yaw = self.yaw + 0.5 * yaw_delta

        self.x += distance * math.cos(mid_yaw)
        self.y += distance * math.sin(mid_yaw)
        self.yaw = _wrap(self.yaw + yaw_delta)

    def _record_history(self, time_sec):
        self.history.append((time_sec, self.x, self.y, self.yaw))
        cutoff = time_sec - 1.5
        self.history = [entry for entry in self.history if entry[0] >= cutoff]

    def _fuse_april_tag(self, measurement):
        if measurement.ambiguity > 0.35 or not self.history:
            return

        reference = min(self.history, key=lambda entry: abs(entry[0] - measurement.timestamp))
        _, ref_x, ref_y, ref_yaw = reference
        position_error_x = measurement.pose.x - ref_x
        position_error_y = measurement.pose.y - ref_y
        yaw_error = _wrap(measurement.pose.yaw - ref_yaw)

        quality = max(0.15, min(1.0, 1.0 - 2.2 * measurement.ambiguity))
        position_gain = quality * max(0.08, min(0.48, 0.030 / max(0.018, measurement.position_std_m)))
        yaw_gain = quality * max(0.05, min(0.32, 0.035 / max(0.030, measurement.yaw_std_rad)))
        dx = position_gain * position_error_x
        dy = position_gain * position_error_y
        dyaw = yaw_gain * yaw_error

        self.x += dx
        self.y += dy
        self.yaw = _wrap(self.yaw + dyaw)

        corrected = []
        for time_sec, x, y, yaw in self.history:
            if time_sec >= measurement.timestamp - 1e-6:
                corrected.append((time_sec, x + dx, y + dy, _wrap(yaw + dyaw)))
            else:
                corrected.append((time_sec, x, y, yaw))
        self.history = corrected
