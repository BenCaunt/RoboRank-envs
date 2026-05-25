import math


class RobotPolicy:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_left_ticks = None
        self.last_right_ticks = None

    def step(self, robot):
        left_ticks, right_ticks = robot.get_encoder_values()
        if self.last_left_ticks is None:
            self.last_left_ticks = left_ticks
            self.last_right_ticks = right_ticks

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
        self.yaw = math.atan2(math.sin(self.yaw + yaw_delta), math.cos(self.yaw + yaw_delta))
        robot.submit_odometry(Pose2d(self.x, self.y, self.yaw))
