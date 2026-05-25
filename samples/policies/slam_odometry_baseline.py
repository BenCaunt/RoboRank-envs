import math


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class RobotPolicy:
    def __init__(self):
        self.prev_ticks = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.map_cells = {}

    def step(self, robot):
        left_ticks, right_ticks = robot.get_encoder_values()
        if self.prev_ticks is None:
            self.prev_ticks = (left_ticks, right_ticks)

        prev_left, prev_right = self.prev_ticks
        self.prev_ticks = (left_ticks, right_ticks)
        meters_per_tick = 2.0 * math.pi * robot.wheel_radius_m / robot.ticks_per_rev
        left_delta = (left_ticks - prev_left) * meters_per_tick
        right_delta = (right_ticks - prev_right) * meters_per_tick
        distance = 0.5 * (left_delta + right_delta)
        yaw_delta = robot.gyro() * robot.dt
        mid_yaw = self.yaw + 0.5 * yaw_delta

        self.x += distance * math.cos(mid_yaw)
        self.y += distance * math.sin(mid_yaw)
        self.yaw = _wrap(self.yaw + yaw_delta)

        self._integrate_scan(robot, self.x, self.y, self.yaw)
        robot.submit_slam(
            Pose2d(self.x, self.y, self.yaw),
            [MapPoint2d(x, y) for x, y in list(self.map_cells.values())[: robot.max_map_points]],
        )

    def _integrate_scan(self, robot, pose_x, pose_y, pose_yaw):
        scan = robot.lidar()
        angles = robot.lidar_angles()
        stride = 3
        for index in range(0, len(scan), stride):
            distance = float(scan[index])
            if distance <= 0.10 or distance >= robot.lidar_max_range_m - 0.05:
                continue
            angle = pose_yaw + float(angles[index])
            x = pose_x + distance * math.cos(angle)
            y = pose_y + distance * math.sin(angle)
            key = (round(x / 0.08), round(y / 0.08))
            if key not in self.map_cells and len(self.map_cells) < robot.max_map_points:
                self.map_cells[key] = (x, y)
