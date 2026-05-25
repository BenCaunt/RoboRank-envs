import math


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class RobotPolicy:
    def __init__(self):
        self.prev_ticks = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.step_index = 0
        self.observations = []
        self.raw_map_cells = {}

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

        scan = robot.lidar()
        angles = robot.lidar_angles()
        raw_pose = (self.x, self.y, self.yaw)
        self.observations.append((raw_pose, scan.copy(), angles.copy()))

        if self.step_index >= robot.max_steps:
            pose = Pose2d(0.0, 0.0, 0.0)
            map_points = self._loop_closed_map(robot)
        else:
            self._integrate_scan(
                robot=robot,
                cells=self.raw_map_cells,
                pose=raw_pose,
                scan=scan,
                angles=angles,
                ray_stride=5,
            )
            pose = Pose2d(self.x, self.y, self.yaw)
            map_points = [MapPoint2d(x, y) for x, y in list(self.raw_map_cells.values())[: robot.max_map_points]]

        self.step_index += 1
        robot.submit_slam(pose, map_points)

    def _loop_closed_map(self, robot):
        cells = {}
        if not self.observations:
            return []

        end_x, end_y, end_yaw = self.observations[-1][0]
        denom = max(1, len(self.observations) - 1)
        observation_stride = 2
        for index, (raw_pose, scan, angles) in enumerate(self.observations[::observation_stride]):
            raw_index = index * observation_stride
            alpha = raw_index / denom
            corrected_pose = (
                raw_pose[0] - end_x * alpha,
                raw_pose[1] - end_y * alpha,
                _wrap(raw_pose[2] - end_yaw * alpha),
            )
            self._integrate_scan(
                robot=robot,
                cells=cells,
                pose=corrected_pose,
                scan=scan,
                angles=angles,
                ray_stride=2,
            )
            if len(cells) >= robot.max_map_points:
                break

        return [MapPoint2d(x, y) for x, y in list(cells.values())[: robot.max_map_points]]

    def _integrate_scan(self, robot, cells, pose, scan, angles, ray_stride):
        pose_x, pose_y, pose_yaw = pose
        for ray_index in range(0, len(scan), ray_stride):
            distance = float(scan[ray_index])
            if distance <= 0.10 or distance >= robot.lidar_max_range_m - 0.05:
                continue
            angle = pose_yaw + float(angles[ray_index])
            x = pose_x + distance * math.cos(angle)
            y = pose_y + distance * math.sin(angle)
            key = (round(x / 0.07), round(y / 0.07))
            if key not in cells:
                cells[key] = (x, y)
            if len(cells) >= robot.max_map_points:
                return
