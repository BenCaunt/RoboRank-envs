from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from roborank_envs.catalog import DIFF_DRIVE_REACH_TARGET
from roborank_envs.models import (
    AprilTagPoseSample,
    ChargerPoseSample,
    ChallengeSpec,
    CollisionEvent,
    ControlSample,
    EncoderTraceSample,
    GyroTraceSample,
    LidarScanSample,
    MapPoint2D,
    Obstacle,
    OdometryEstimateSample,
    Pose2D,
    PoseSample,
    ReplayTrace,
    RenderFrame,
    RunResult,
    ScoreMetrics,
    Target,
)
from roborank_envs.policy_api import (
    AprilTagPoseEstimate as ApiAprilTagPoseEstimate,
    CircleObstacle,
    DifferentialDrive as DifferentialDriveApi,
    DifferentialDriveOdometry as DifferentialDriveOdometryApi,
    DifferentialDriveSlam as DifferentialDriveSlamApi,
    DifferentialDriveStateEstimator as DifferentialDriveStateEstimatorApi,
    MapPoint2d as ApiMapPoint2d,
    Pose2d as ApiPose2d,
    RobotPolicyProtocol,
    Target2d as ApiTarget2d,
)
from roborank_envs.simulation.mujoco_world import MujocoWorld
from roborank_envs.simulation.rendering import configured_render_interval_sec
from roborank_envs.simulation.rerun_export import write_rerun_recording

DEFAULT_REPLAY_RENDER_INTERVAL_SEC = 0.1
DOCK_TO_CHARGER_SCENARIO = "dock_to_charger"
DIFF_DRIVE_STATE_ESTIMATION_RUNNER = "diff_drive_state_estimation"
DIFF_DRIVE_SLAM_RUNNER = "diff_drive_slam"
CHARGER_CAMERA_WIDTH = 160
CHARGER_CAMERA_HEIGHT = 120
CHARGER_CAMERA_FOV_RAD = math.radians(120.0)
APRIL_TAG_CAMERA_FOV_RAD = math.radians(120.0)


@dataclass(frozen=True)
class AprilTagLandmark:
    tag_id: int
    x: float
    y: float
    yaw: float
    size_m: float = 0.165


@dataclass(frozen=True)
class PendingAprilTagMeasurement:
    delivery_step: int
    measurement: ApiAprilTagPoseEstimate
    sample: AprilTagPoseSample


@dataclass(frozen=True)
class ArenaBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class Scenario:
    start: Pose2D
    target: Target
    route: list[Pose2D]
    obstacles: list[Obstacle]
    bounds: ArenaBounds
    target_yaw: float = 0.0
    sensor_seed: int = 0
    charger_dropout_windows: tuple[tuple[float, float], ...] = ()
    charger_position_noise_m: float = 0.0
    charger_yaw_noise_rad: float = 0.0
    charger_bias_x_m: float = 0.0
    charger_bias_y_m: float = 0.0
    charger_bias_yaw_rad: float = 0.0
    april_tags: tuple[AprilTagLandmark, ...] = ()


@dataclass(frozen=True)
class OdometrySensorState:
    left_ticks: int
    right_ticks: int
    gyro_z_radps: float


class SimulationError(RuntimeError):
    pass


class DifferentialDriveRunner:
    def __init__(self, challenge: ChallengeSpec = DIFF_DRIVE_REACH_TARGET) -> None:
        self.challenge = challenge
        self.dt = float(challenge.defaults["dt_sec"])
        self.default_max_steps = int(challenge.defaults["max_steps"])
        self.render_interval_sec = configured_render_interval_sec(
            float(challenge.defaults.get("render_interval_sec", DEFAULT_REPLAY_RENDER_INTERVAL_SEC))
        )
        bounds = challenge.defaults["arena_bounds_m"]
        self.bounds = ArenaBounds(
            x_min=float(bounds["x_min"]),
            x_max=float(bounds["x_max"]),
            y_min=float(bounds["y_min"]),
            y_max=float(bounds["y_max"]),
        )

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        max_step_count = max_steps or self.default_max_steps
        scenario = self._scenario(seed)
        robot = self.challenge.robot
        world = MujocoWorld.create(
            target=scenario.target,
            obstacles=scenario.obstacles,
            bounds=scenario.bounds.__dict__,
            robot=robot,
            target_yaw=scenario.target_yaw,
            target_kind="charging_pad" if self._is_docking_challenge() else "target_disk",
        )

        pose = scenario.start
        initial_distance = _distance_to_target(pose, scenario.target)
        poses = [PoseSample(t=0.0, x=pose.x, y=pose.y, yaw=pose.yaw)]
        controls: list[ControlSample] = []
        collisions: list[CollisionEvent] = []
        render_frames: list[RenderFrame] = []
        lidar_scans: list[LidarScanSample] = []
        charger_pose_samples: list[ChargerPoseSample] = []
        render_error: str | None = None
        rerun_error: str | None = None
        path_length = 0.0
        energy_used = 0.0
        smoothness_cost = 0.0
        last_control: tuple[float, float] | None = None
        left_encoder_ticks = 0.0
        right_encoder_ticks = 0.0
        gyro_z = 0.0
        status = "timeout"
        final_step = 0

        world.reset_robot_pose(pose)
        policy_robot = DifferentialDriveApi(
            wheel_base_m=robot.wheel_base_m,
            max_wheel_velocity_mps=robot.max_wheel_velocity_mps,
            dt=self.dt,
            max_steps=max_step_count,
            seed=seed,
            wheel_radius_m=robot.wheel_radius_m,
            ticks_per_rev=int(self.challenge.defaults.get("ticks_per_rev", 392)),
        )
        self._update_policy_robot(
            robot=policy_robot,
            t=0.0,
            pose=pose,
            scenario=scenario,
            collision_count=0,
            encoder_values=(left_encoder_ticks, right_encoder_ticks),
            gyro_z=gyro_z,
        )

        try:
            render_stride = max(1, round(self.render_interval_sec / self.dt))
            self._append_lidar_scan(scans=lidar_scans, t=0.0, frame_index=0, pose=pose, scenario=scenario)
            render_error = self._append_render_frames(
                world=world,
                frames=render_frames,
                t=0.0,
                frame_index=0,
                render_error=render_error,
            )

            for step in range(max_step_count):
                t = step * self.dt
                self._update_policy_robot(
                    robot=policy_robot,
                    t=t,
                    pose=pose,
                    scenario=scenario,
                    collision_count=len(collisions),
                    encoder_values=(left_encoder_ticks, right_encoder_ticks),
                    gyro_z=gyro_z,
                    charger_pose_samples=charger_pose_samples,
                )
                left, right = self._call_policy_step(policy, policy_robot)
                controls.append(
                    ControlSample(
                        t=round(t, 6),
                        left_wheel_velocity=left,
                        right_wheel_velocity=right,
                    )
                )

                if last_control is not None:
                    smoothness_cost += ((left - last_control[0]) ** 2 + (right - last_control[1]) ** 2) * self.dt
                last_control = (left, right)
                energy_used += (left**2 + right**2) * self.dt
                left_encoder_ticks += _wheel_velocity_to_ticks(
                    velocity=left,
                    dt=self.dt,
                    wheel_radius=robot.wheel_radius_m,
                    ticks_per_rev=policy_robot.ticks_per_rev,
                )
                right_encoder_ticks += _wheel_velocity_to_ticks(
                    velocity=right,
                    dt=self.dt,
                    wheel_radius=robot.wheel_radius_m,
                    ticks_per_rev=policy_robot.ticks_per_rev,
                )
                gyro_z = (right - left) / robot.wheel_base_m

                next_pose = (
                    world.step_diff_drive(left=left, right=right, wheel_base=robot.wheel_base_m, dt=self.dt)
                    if world.available
                    else self._integrate(pose, left=left, right=right)
                )
                path_length += math.hypot(next_pose.x - pose.x, next_pose.y - pose.y)
                pose = next_pose
                final_step = step + 1

                poses.append(PoseSample(t=round(final_step * self.dt, 6), x=pose.x, y=pose.y, yaw=pose.yaw))
                if final_step % render_stride == 0:
                    self._append_lidar_scan(
                        scans=lidar_scans,
                        t=final_step * self.dt,
                        frame_index=final_step,
                        pose=pose,
                        scenario=scenario,
                    )
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=final_step * self.dt,
                        frame_index=final_step,
                        render_error=render_error,
                    )

                collision = (
                    self._mujoco_collision(t=(step + 1) * self.dt, world=world)
                    if world.available
                    else self._first_collision(t=(step + 1) * self.dt, pose=pose, scenario=scenario)
                )
                analytic_collision = self._first_collision(t=(step + 1) * self.dt, pose=pose, scenario=scenario)
                if collision is None and analytic_collision is not None:
                    collision = analytic_collision
                if collision is not None:
                    collisions.append(collision)
                    status = "collision"
                    self._append_lidar_scan(
                        scans=lidar_scans,
                        t=final_step * self.dt,
                        frame_index=final_step,
                        pose=pose,
                        scenario=scenario,
                    )
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=final_step * self.dt,
                        frame_index=final_step,
                        render_error=render_error,
                    )
                    break

                if self._has_reached_goal(pose=pose, scenario=scenario):
                    status = "success"
                    self._append_lidar_scan(
                        scans=lidar_scans,
                        t=final_step * self.dt,
                        frame_index=final_step,
                        pose=pose,
                        scenario=scenario,
                    )
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=final_step * self.dt,
                        frame_index=final_step,
                        render_error=render_error,
                    )
                    break
        finally:
            world.close()

        final_distance = _distance_to_target(pose, scenario.target)
        final_heading_error = abs(_wrap_angle(scenario.target_yaw - pose.yaw))
        elapsed = final_step * self.dt
        metrics = self._score(
            status=status,
            elapsed=elapsed,
            final_distance=final_distance,
            final_heading_error=final_heading_error,
            initial_distance=initial_distance,
            collisions=len(collisions),
            path_length=path_length,
            energy_used=energy_used,
            smoothness_cost=smoothness_cost,
        )
        replay = ReplayTrace(
            poses=poses,
            target=scenario.target,
            obstacles=scenario.obstacles,
            collisions=collisions,
            controls=controls,
            render_frames=render_frames,
            lidar_scans=lidar_scans,
            charger_pose_samples=charger_pose_samples,
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "challenge_mode": self.challenge.defaults.get("scenario", "reach_target"),
                "mujoco_backend": world.backend_name,
                "simulation_mode": "stepped_mujoco_physics" if world.available else "kinematic_fallback",
                "mujoco_timestep_sec": world.timestep if world.available else None,
                "mujoco_render_frames": len(render_frames),
                "mujoco_render_cameras": list(world.render_cameras),
                "mujoco_render_interval_sec": self.render_interval_sec if world.available else None,
                "mujoco_render_resolution": {
                    "width": world.render_width,
                    "height": world.render_height,
                },
                "mujoco_render_error": render_error,
                "robot_model": robot.model,
                "robot_dimensions_m": {
                    "length": robot.footprint_length_m,
                    "width": robot.footprint_width_m,
                    "height": robot.height_m,
                },
                "route_waypoints": [waypoint.model_dump() for waypoint in scenario.route],
                "target_yaw_rad": scenario.target_yaw,
                "lidar_scan_count": len(lidar_scans),
                "charger_pose_sample_count": len(charger_pose_samples),
                "arena_bounds_m": scenario.bounds.__dict__,
            },
        )
        artifact, rerun_error = write_rerun_recording(
            challenge=self.challenge,
            seed=seed,
            replay=replay,
            metrics=metrics,
        )
        if artifact is not None:
            replay.artifacts.append(artifact)
        replay.metadata["rerun_artifacts"] = len(replay.artifacts)
        replay.metadata["rerun_export_error"] = rerun_error
        return RunResult(challenge_id=self.challenge.id, seed=seed, metrics=metrics, replay=replay)

    def _scenario(self, seed: int) -> Scenario:
        if self.challenge.defaults.get("scenario") == DOCK_TO_CHARGER_SCENARIO:
            return self._dock_to_charger_scenario(seed)
        if self.challenge.defaults.get("scenario") == "warehouse_aisle_avoidance":
            return self._warehouse_aisle_scenario(seed)
        return self._reach_target_scenario(seed)

    def _dock_to_charger_scenario(self, seed: int) -> Scenario:
        rng = random.Random(seed)
        target = Target(
            x=1.15 + rng.uniform(-0.04, 0.04),
            y=rng.uniform(-0.03, 0.03),
            radius=float(self.challenge.success_conditions["position_tolerance_m"]),
        )
        target_yaw = rng.uniform(-0.035, 0.035)
        start = Pose2D(
            x=-1.48 + rng.uniform(-0.16, 0.10),
            y=target.y + rng.uniform(-0.42, 0.42),
            yaw=target_yaw + rng.uniform(-0.34, 0.34),
        )
        obstacles = [
            Obstacle(id="left_dock_bumper", x=target.x + 0.03, y=target.y + 0.44, radius=0.10),
            Obstacle(id="right_dock_bumper", x=target.x + 0.03, y=target.y - 0.44, radius=0.10),
            Obstacle(id="cable_guard", x=0.10 + rng.uniform(-0.04, 0.04), y=0.78, radius=0.12),
            Obstacle(id="maintenance_stand", x=-0.35 + rng.uniform(-0.04, 0.04), y=-0.82, radius=0.14),
        ]
        route = [
            Pose2D(x=start.x, y=start.y, yaw=start.yaw),
            Pose2D(x=-0.55, y=target.y * 0.5, yaw=target_yaw),
            Pose2D(x=0.35, y=target.y, yaw=target_yaw),
            Pose2D(x=target.x, y=target.y, yaw=target_yaw),
        ]
        dropout_start = 4.0 + rng.uniform(-0.35, 0.35)
        return Scenario(
            start=start,
            target=target,
            route=route,
            obstacles=obstacles,
            bounds=self.bounds,
            target_yaw=target_yaw,
            sensor_seed=seed,
            charger_dropout_windows=((dropout_start, dropout_start + 0.28),),
            charger_position_noise_m=0.012,
            charger_yaw_noise_rad=math.radians(0.7),
            charger_bias_x_m=rng.uniform(-0.006, 0.006),
            charger_bias_y_m=rng.uniform(-0.006, 0.006),
            charger_bias_yaw_rad=rng.uniform(-math.radians(0.35), math.radians(0.35)),
        )

    def _reach_target_scenario(self, seed: int) -> Scenario:
        rng = random.Random(seed)
        start = Pose2D(
            x=-1.85 + rng.uniform(-0.12, 0.12),
            y=-1.15 + rng.uniform(-0.10, 0.10),
            yaw=rng.uniform(-0.25, 0.25),
        )
        target = Target(
            x=1.75 + rng.uniform(-0.10, 0.10),
            y=1.15 + rng.uniform(-0.10, 0.10),
            radius=float(self.challenge.success_conditions["target_tolerance_m"]),
        )
        obstacles = [
            Obstacle(id="upper_left_post", x=-0.65 + rng.uniform(-0.05, 0.05), y=0.85, radius=0.24),
            Obstacle(id="lower_right_post", x=0.85 + rng.uniform(-0.05, 0.05), y=-0.65, radius=0.28),
        ]
        route = [Pose2D(x=start.x, y=start.y, yaw=start.yaw), Pose2D(x=target.x, y=target.y, yaw=0.0)]
        return Scenario(start=start, target=target, route=route, obstacles=obstacles, bounds=self.bounds)

    def _warehouse_aisle_scenario(self, seed: int) -> Scenario:
        rng = random.Random(seed)
        start = Pose2D(
            x=-2.78 + rng.uniform(-0.06, 0.06),
            y=rng.uniform(-0.08, 0.08),
            yaw=rng.uniform(-0.08, 0.08),
        )
        target = Target(
            x=2.78 + rng.uniform(-0.06, 0.06),
            y=rng.uniform(-0.06, 0.06),
            radius=float(self.challenge.success_conditions["target_tolerance_m"]),
        )

        upper_y = 1.08 + rng.uniform(-0.02, 0.02)
        lower_y = -1.08 + rng.uniform(-0.02, 0.02)
        post_xs = [-2.55, -1.75, -0.95, -0.15, 0.65, 1.45, 2.25]
        obstacles: list[Obstacle] = []
        for index, x in enumerate(post_xs):
            offset = rng.uniform(-0.035, 0.035)
            obstacles.append(Obstacle(id=f"upper_rack_post_{index + 1}", x=x + offset, y=upper_y, radius=0.18))
            obstacles.append(Obstacle(id=f"lower_rack_post_{index + 1}", x=x - offset, y=lower_y, radius=0.18))

        south_pallet = Obstacle(
            id="south_pallet_intrusion",
            x=-0.92 + rng.uniform(-0.06, 0.06),
            y=-0.30 + rng.uniform(-0.04, 0.04),
            radius=0.30,
        )
        north_pallet = Obstacle(
            id="north_pallet_intrusion",
            x=0.76 + rng.uniform(-0.06, 0.06),
            y=0.32 + rng.uniform(-0.04, 0.04),
            radius=0.30,
        )
        staging_cone = Obstacle(
            id="staging_cone",
            x=1.85 + rng.uniform(-0.04, 0.04),
            y=0.58 + rng.uniform(-0.03, 0.03),
            radius=0.18,
        )
        obstacles.extend([south_pallet, north_pallet, staging_cone])

        route = [
            Pose2D(x=start.x, y=start.y, yaw=start.yaw),
            Pose2D(x=-2.10, y=0.18, yaw=0.0),
            Pose2D(x=-1.42, y=0.56, yaw=0.0),
            Pose2D(x=-0.25, y=0.56, yaw=0.0),
            Pose2D(x=0.55, y=-0.52, yaw=0.0),
            Pose2D(x=1.70, y=-0.34, yaw=0.0),
            Pose2D(x=target.x, y=target.y, yaw=0.0),
        ]

        return Scenario(start=start, target=target, route=route, obstacles=obstacles, bounds=self.bounds)

    def _update_policy_robot(
        self,
        *,
        robot: DifferentialDriveApi,
        t: float,
        pose: Pose2D,
        scenario: Scenario,
        collision_count: int,
        encoder_values: tuple[float, float] = (0.0, 0.0),
        gyro_z: float = 0.0,
        charger_pose_samples: list[ChargerPoseSample] | None = None,
    ) -> None:
        frame_index = round(t / self.dt)
        lidar_scan = self._lidar_scan(t=t, frame_index=frame_index, pose=pose, scenario=scenario)
        charger_pose: ApiPose2d | None = None
        camera_frame: np.ndarray | None = None
        docking_observation = self._docking_observation(
            t=t,
            frame_index=frame_index,
            pose=pose,
            scenario=scenario,
        )
        if docking_observation is not None:
            charger_pose, camera_frame, sample = docking_observation
            if charger_pose_samples is not None and all(
                existing.frame_index != sample.frame_index for existing in charger_pose_samples
            ):
                charger_pose_samples.append(sample)

        robot._update(
            time=round(t, 6),
            pose=ApiPose2d(x=pose.x, y=pose.y, yaw=pose.yaw),
            target=ApiTarget2d(x=scenario.target.x, y=scenario.target.y, radius=scenario.target.radius),
            route=tuple(ApiPose2d(x=waypoint.x, y=waypoint.y, yaw=waypoint.yaw) for waypoint in scenario.route),
            obstacles=tuple(
                CircleObstacle(id=obstacle.id, x=obstacle.x, y=obstacle.y, radius=obstacle.radius)
                for obstacle in scenario.obstacles
            ),
            lidar_ranges=np.asarray(lidar_scan.ranges_m, dtype=float),
            collision_count=collision_count,
            encoder_values=encoder_values,
            gyro_z=gyro_z,
            charger_pose=charger_pose,
            camera_frame=camera_frame,
        )

    def _docking_observation(
        self,
        *,
        t: float,
        frame_index: int,
        pose: Pose2D,
        scenario: Scenario,
    ) -> tuple[ApiPose2d | None, np.ndarray, ChargerPoseSample] | None:
        if not self._is_docking_challenge():
            return None

        rel_x, rel_y, rel_yaw = _relative_pose(pose=pose, target=scenario.target, target_yaw=scenario.target_yaw)
        distance = math.hypot(rel_x, rel_y)
        bearing = math.atan2(rel_y, max(rel_x, 1e-6))
        in_dropout = any(start <= t <= end for start, end in scenario.charger_dropout_windows)
        visible = (
            rel_x > -0.08
            and distance <= 3.2
            and abs(bearing) <= CHARGER_CAMERA_FOV_RAD * 0.48
            and not in_dropout
        )

        estimate: ApiPose2d | None = None
        sample = ChargerPoseSample(t=round(t, 6), frame_index=frame_index, visible=visible)
        rng = random.Random((scenario.sensor_seed + 1) * 1_000_003 + frame_index * 9_176 + 17)
        if visible:
            estimated_x = rel_x + scenario.charger_bias_x_m + rng.gauss(0.0, scenario.charger_position_noise_m)
            estimated_y = rel_y + scenario.charger_bias_y_m + rng.gauss(0.0, scenario.charger_position_noise_m)
            estimated_yaw = _wrap_angle(
                rel_yaw + scenario.charger_bias_yaw_rad + rng.gauss(0.0, scenario.charger_yaw_noise_rad)
            )
            estimate = ApiPose2d(x=estimated_x, y=estimated_y, yaw=estimated_yaw)
            sample = ChargerPoseSample(
                t=round(t, 6),
                frame_index=frame_index,
                visible=True,
                x=round(estimated_x, 6),
                y=round(estimated_y, 6),
                yaw=round(estimated_yaw, 6),
            )

        return estimate, _charger_camera_frame(rel_x=rel_x, rel_y=rel_y, visible=visible), sample

    def _call_policy_step(self, policy: RobotPolicyProtocol, robot: DifferentialDriveApi) -> tuple[float, float]:
        try:
            robot._clear_command()
            step_result = policy.step(robot)
            if step_result is not None:
                raise SimulationError(
                    "RobotPolicy.step(robot) should command robot.set_wheel_velocity(left, right), not return an action."
                )
            return robot._consume_wheel_command()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _integrate(self, pose: Pose2D, *, left: float, right: float) -> Pose2D:
        wheel_base = self.challenge.robot.wheel_base_m
        linear_velocity = 0.5 * (left + right)
        angular_velocity = (right - left) / wheel_base

        if abs(angular_velocity) < 1e-9:
            dx = linear_velocity * math.cos(pose.yaw) * self.dt
            dy = linear_velocity * math.sin(pose.yaw) * self.dt
            yaw = pose.yaw
        else:
            yaw = pose.yaw + angular_velocity * self.dt
            radius = linear_velocity / angular_velocity
            dx = radius * (math.sin(yaw) - math.sin(pose.yaw))
            dy = -radius * (math.cos(yaw) - math.cos(pose.yaw))

        return Pose2D(x=pose.x + dx, y=pose.y + dy, yaw=_wrap_angle(yaw))

    def _first_collision(self, *, t: float, pose: Pose2D, scenario: Scenario) -> CollisionEvent | None:
        robot_radius = self.challenge.robot.radius_m
        bounds = scenario.bounds
        bound_penetration = max(
            bounds.x_min + robot_radius - pose.x,
            pose.x - (bounds.x_max - robot_radius),
            bounds.y_min + robot_radius - pose.y,
            pose.y - (bounds.y_max - robot_radius),
        )
        if bound_penetration > 0.0:
            return CollisionEvent(
                t=round(t, 6),
                kind="bounds",
                object_id="arena_bounds",
                penetration_m=round(bound_penetration, 6),
            )

        for obstacle in scenario.obstacles:
            required_clearance = robot_radius + obstacle.radius
            distance = math.hypot(pose.x - obstacle.x, pose.y - obstacle.y)
            penetration = required_clearance - distance
            if penetration > 0.0:
                return CollisionEvent(
                    t=round(t, 6),
                    kind="obstacle",
                    object_id=obstacle.id,
                    penetration_m=round(penetration, 6),
                )

        return None

    def _mujoco_collision(self, *, t: float, world: MujocoWorld) -> CollisionEvent | None:
        contact = world.first_contact()
        if contact is None:
            return None
        return CollisionEvent(
            t=round(t, 6),
            kind=contact.kind,  # type: ignore[arg-type]
            object_id=contact.object_id,
            penetration_m=round(contact.penetration_m, 6),
        )

    def _append_render_frames(
        self,
        *,
        world: MujocoWorld,
        frames: list[RenderFrame],
        t: float,
        frame_index: int,
        render_error: str | None,
    ) -> str | None:
        if not world.available or render_error is not None:
            return render_error

        for camera in world.render_cameras:
            if any(frame.frame_index == frame_index and frame.camera == camera for frame in frames):
                continue

            try:
                frames.append(
                    RenderFrame(
                        t=round(t, 6),
                        frame_index=frame_index,
                        image_data_url=world.render_data_url(camera=camera),
                        width=world.render_width,
                        height=world.render_height,
                        camera=camera,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - rendering should not break grading.
                return f"{type(exc).__name__}: {exc}"
        return None

    def _append_lidar_scan(
        self,
        *,
        scans: list[LidarScanSample],
        t: float,
        frame_index: int,
        pose: Pose2D,
        scenario: Scenario,
    ) -> None:
        if any(scan.frame_index == frame_index for scan in scans):
            return
        scans.append(self._lidar_scan(t=t, frame_index=frame_index, pose=pose, scenario=scenario))

    def _lidar_scan(self, *, t: float, frame_index: int, pose: Pose2D, scenario: Scenario) -> LidarScanSample:
        lidar = self.challenge.robot.lidars[0] if self.challenge.robot.lidars else None
        if lidar is None:
            return LidarScanSample(
                t=round(t, 6),
                frame_index=frame_index,
                frame="base_link",
                angles_rad=[],
                ranges_m=[],
                max_range_m=0.0,
            )

        ray_count = max(1, lidar.num_rays)
        if ray_count == 1:
            angles = [0.0]
        else:
            fov_rad = math.radians(lidar.fov_deg)
            angles = [-fov_rad / 2 + index * fov_rad / ray_count for index in range(ray_count)]

        origin_x = pose.x + math.cos(pose.yaw) * lidar.xyz_m.get("x", 0.0) - math.sin(pose.yaw) * lidar.xyz_m.get("y", 0.0)
        origin_y = pose.y + math.sin(pose.yaw) * lidar.xyz_m.get("x", 0.0) + math.cos(pose.yaw) * lidar.xyz_m.get("y", 0.0)
        ranges = [
            round(
                self._ray_range(
                    origin_x=origin_x,
                    origin_y=origin_y,
                    angle=pose.yaw + angle,
                    max_range=lidar.max_range_m,
                    scenario=scenario,
                ),
                4,
            )
            for angle in angles
        ]

        return LidarScanSample(
            t=round(t, 6),
            frame_index=frame_index,
            frame=lidar.frame,
            angles_rad=[round(angle, 6) for angle in angles],
            ranges_m=ranges,
            max_range_m=lidar.max_range_m,
        )

    def _ray_range(
        self,
        *,
        origin_x: float,
        origin_y: float,
        angle: float,
        max_range: float,
        scenario: Scenario,
    ) -> float:
        dx = math.cos(angle)
        dy = math.sin(angle)
        best = max_range

        bounds = scenario.bounds
        candidates: list[float] = []
        if abs(dx) > 1e-9:
            candidates.extend(((bounds.x_min - origin_x) / dx, (bounds.x_max - origin_x) / dx))
        if abs(dy) > 1e-9:
            candidates.extend(((bounds.y_min - origin_y) / dy, (bounds.y_max - origin_y) / dy))

        for distance in candidates:
            if distance <= 0.0 or distance >= best:
                continue
            hit_x = origin_x + distance * dx
            hit_y = origin_y + distance * dy
            if bounds.x_min - 1e-9 <= hit_x <= bounds.x_max + 1e-9 and bounds.y_min - 1e-9 <= hit_y <= bounds.y_max + 1e-9:
                best = distance

        for obstacle in scenario.obstacles:
            ox = origin_x - obstacle.x
            oy = origin_y - obstacle.y
            b = 2.0 * (dx * ox + dy * oy)
            c = ox * ox + oy * oy - obstacle.radius * obstacle.radius
            discriminant = b * b - 4.0 * c
            if discriminant < 0.0:
                continue
            root = math.sqrt(discriminant)
            for distance in ((-b - root) / 2.0, (-b + root) / 2.0):
                if 0.0 < distance < best:
                    best = distance

        return max(0.0, min(max_range, best))

    def _score(
        self,
        *,
        status: str,
        elapsed: float,
        final_distance: float,
        final_heading_error: float,
        initial_distance: float,
        collisions: int,
        path_length: float,
        energy_used: float,
        smoothness_cost: float,
    ) -> ScoreMetrics:
        max_time = self.default_max_steps * self.dt
        progress = max(0.0, min(1.0, 1.0 - final_distance / max(initial_distance, 1e-9)))
        success = status == "success" and collisions == 0

        if self._is_docking_challenge():
            yaw_tolerance = float(self.challenge.success_conditions["yaw_tolerance_rad"])
            alignment = max(0.0, min(1.0, 1.0 - final_heading_error / max(yaw_tolerance * 6.0, 1e-9)))
            success_points = 60.0 if success else 38.0 * progress
            alignment_points = 15.0 * alignment
            time_points = 10.0 * max(0.0, 1.0 - elapsed / max_time) if success else 0.0
            safety_points = max(0.0, 10.0 - 8.0 * collisions)
            smooth_energy_points = max(0.0, 5.0 - 0.35 * smoothness_cost - 0.04 * energy_used)
            total = min(100.0, success_points + alignment_points + time_points + safety_points + smooth_energy_points)
            return ScoreMetrics(
                score=round(total, 3),
                success=success,
                status=status,  # type: ignore[arg-type]
                elapsed_sec=round(elapsed, 6),
                distance_to_target_m=round(final_distance, 6),
                heading_error_rad=round(final_heading_error, 6),
                collision_count=collisions,
                path_length_m=round(path_length, 6),
                energy_used=round(energy_used, 6),
                smoothness_cost=round(smoothness_cost, 6),
            )

        success_points = 55.0 if success else 35.0 * progress
        time_points = 20.0 * max(0.0, 1.0 - elapsed / max_time) if success else 0.0
        safety_points = max(0.0, 15.0 - 7.5 * collisions)
        smoothness_points = max(0.0, 5.0 - 0.4 * smoothness_cost)
        energy_points = max(0.0, 5.0 - 0.08 * energy_used)
        total = min(100.0, success_points + time_points + safety_points + smoothness_points + energy_points)

        return ScoreMetrics(
            score=round(total, 3),
            success=success,
            status=status,  # type: ignore[arg-type]
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(final_distance, 6),
            heading_error_rad=round(final_heading_error, 6),
            collision_count=collisions,
            path_length_m=round(path_length, 6),
            energy_used=round(energy_used, 6),
            smoothness_cost=round(smoothness_cost, 6),
        )

    def _has_reached_goal(self, *, pose: Pose2D, scenario: Scenario) -> bool:
        if self._is_docking_challenge():
            return (
                _distance_to_target(pose, scenario.target)
                <= float(self.challenge.success_conditions["position_tolerance_m"])
                and abs(_wrap_angle(scenario.target_yaw - pose.yaw))
                <= float(self.challenge.success_conditions["yaw_tolerance_rad"])
            )
        return _distance_to_target(pose, scenario.target) <= scenario.target.radius

    def _is_docking_challenge(self) -> bool:
        return self.challenge.defaults.get("scenario") == DOCK_TO_CHARGER_SCENARIO


class DifferentialDriveOdometryRunner(DifferentialDriveRunner):
    """Differential-drive runner that grades submitted odometry estimates."""

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        max_step_count = max_steps or self.default_max_steps
        state_estimation_mode = self._is_state_estimation_challenge()
        scenario = self._state_estimation_scenario() if state_estimation_mode else self._odometry_scenario()
        robot = self.challenge.robot
        ticks_per_rev = int(self.challenge.defaults.get("ticks_per_rev", 392))
        sensor_rng = np.random.default_rng(seed)
        vision_rng = np.random.default_rng(seed + 17_071)
        parameter_rng = random.Random(seed)
        slip_scale_range = float(self.challenge.defaults.get("wheel_slip_scale_range", 0.0))
        left_slip_scale = 1.0 + parameter_rng.uniform(-slip_scale_range, slip_scale_range)
        right_slip_scale = 1.0 + parameter_rng.uniform(-slip_scale_range, slip_scale_range)
        gyro_bias = parameter_rng.uniform(
            -float(self.challenge.defaults.get("gyro_bias_range_radps", 0.0)),
            float(self.challenge.defaults.get("gyro_bias_range_radps", 0.0)),
        )
        gyro_noise_std = float(self.challenge.defaults.get("gyro_noise_std_radps", 0.0))
        tag_update_stride = max(
            1,
            round(float(self.challenge.defaults.get("april_tag_update_interval_sec", self.dt)) / self.dt),
        )
        tag_latency_steps = max(
            0,
            round(float(self.challenge.defaults.get("april_tag_latency_sec", 0.0)) / self.dt),
        )

        world = MujocoWorld.create(
            target=scenario.target,
            obstacles=scenario.obstacles,
            bounds=scenario.bounds.__dict__,
            robot=robot,
        )

        pose = scenario.start
        poses = [PoseSample(t=0.0, x=pose.x, y=pose.y, yaw=pose.yaw)]
        controls: list[ControlSample] = []
        collisions: list[CollisionEvent] = []
        render_frames: list[RenderFrame] = []
        encoder_samples: list[EncoderTraceSample] = []
        gyro_samples: list[GyroTraceSample] = []
        odometry_estimates: list[OdometryEstimateSample] = []
        april_tag_pose_samples: list[AprilTagPoseSample] = []
        pending_april_tag_measurements: list[PendingAprilTagMeasurement] = []
        render_error: str | None = None
        rerun_error: str | None = None
        path_length = 0.0
        yaw_excitation = 0.0
        energy_used = 0.0
        smoothness_cost = 0.0
        last_control: tuple[float, float] | None = None
        left_encoder_ticks_float = 0.0
        right_encoder_ticks_float = 0.0
        sensor_state = OdometrySensorState(left_ticks=0, right_ticks=0, gyro_z_radps=0.0)
        status = "timeout"
        final_step = 0

        world.reset_robot_pose(pose)
        policy_robot = (
            DifferentialDriveStateEstimatorApi(
                wheel_base_m=robot.wheel_base_m,
                wheel_radius_m=robot.wheel_radius_m,
                max_wheel_velocity_mps=robot.max_wheel_velocity_mps,
                ticks_per_rev=ticks_per_rev,
                dt=self.dt,
                max_steps=max_step_count,
                seed=seed,
            )
            if state_estimation_mode
            else DifferentialDriveOdometryApi(
                wheel_base_m=robot.wheel_base_m,
                wheel_radius_m=robot.wheel_radius_m,
                max_wheel_velocity_mps=robot.max_wheel_velocity_mps,
                ticks_per_rev=ticks_per_rev,
                dt=self.dt,
                max_steps=max_step_count,
                seed=seed,
            )
        )

        try:
            render_stride = max(1, round(self.render_interval_sec / self.dt))
            self._append_odometry_sensor_samples(
                encoder_samples=encoder_samples,
                gyro_samples=gyro_samples,
                t=0.0,
                frame_index=0,
                sensor_state=sensor_state,
            )
            render_error = self._append_render_frames(
                world=world,
                frames=render_frames,
                t=0.0,
                frame_index=0,
                render_error=render_error,
            )
            if state_estimation_mode:
                pending_april_tag_measurements.extend(
                    self._capture_april_tag_measurements(
                        pose=pose,
                        capture_step=0,
                        delivery_step=tag_latency_steps,
                        scenario=scenario,
                        rng=vision_rng,
                    )
                )

            for step in range(max_step_count + 1):
                t = step * self.dt
                april_tag_measurements: tuple[ApiAprilTagPoseEstimate, ...] = ()
                if state_estimation_mode:
                    due_measurements = [
                        item for item in pending_april_tag_measurements if item.delivery_step <= step
                    ]
                    pending_april_tag_measurements = [
                        item for item in pending_april_tag_measurements if item.delivery_step > step
                    ]
                    april_tag_measurements = tuple(item.measurement for item in due_measurements)
                    april_tag_pose_samples.extend(item.sample for item in due_measurements)

                if isinstance(policy_robot, DifferentialDriveStateEstimatorApi):
                    policy_robot._update(
                        time=round(t, 6),
                        encoder_values=(sensor_state.left_ticks, sensor_state.right_ticks),
                        gyro_z=sensor_state.gyro_z_radps,
                        collision_count=len(collisions),
                        april_tag_measurements=april_tag_measurements,
                    )
                else:
                    policy_robot._update(
                        time=round(t, 6),
                        encoder_values=(sensor_state.left_ticks, sensor_state.right_ticks),
                        gyro_z=sensor_state.gyro_z_radps,
                        collision_count=len(collisions),
                    )
                estimate = self._call_odometry_policy_step(policy, policy_robot)
                position_error = math.hypot(estimate.x - pose.x, estimate.y - pose.y)
                yaw_error = abs(_wrap_angle(estimate.yaw - pose.yaw))
                odometry_estimates.append(
                    OdometryEstimateSample(
                        t=round(t, 6),
                        frame_index=step,
                        x=round(estimate.x, 6),
                        y=round(estimate.y, 6),
                        yaw=round(estimate.yaw, 6),
                        position_error_m=round(position_error, 6),
                        yaw_error_rad=round(yaw_error, 6),
                    )
                )

                if step >= max_step_count:
                    final_step = step
                    break

                left, right = self._odometry_trajectory_command(
                    t=t,
                    state_estimation_mode=state_estimation_mode,
                )
                controls.append(
                    ControlSample(
                        t=round(t, 6),
                        left_wheel_velocity=left,
                        right_wheel_velocity=right,
                    )
                )
                if last_control is not None:
                    smoothness_cost += ((left - last_control[0]) ** 2 + (right - last_control[1]) ** 2) * self.dt
                last_control = (left, right)
                energy_used += (left**2 + right**2) * self.dt

                effective_left = left * left_slip_scale
                effective_right = right * right_slip_scale
                next_pose = self._integrate(pose, left=effective_left, right=effective_right)
                path_length += math.hypot(next_pose.x - pose.x, next_pose.y - pose.y)
                yaw_delta = _wrap_angle(next_pose.yaw - pose.yaw)
                yaw_excitation += abs(yaw_delta)

                left_encoder_ticks_float += _wheel_velocity_to_ticks(
                    velocity=left,
                    dt=self.dt,
                    wheel_radius=robot.wheel_radius_m,
                    ticks_per_rev=ticks_per_rev,
                )
                right_encoder_ticks_float += _wheel_velocity_to_ticks(
                    velocity=right,
                    dt=self.dt,
                    wheel_radius=robot.wheel_radius_m,
                    ticks_per_rev=ticks_per_rev,
                )
                sensor_state = OdometrySensorState(
                    left_ticks=int(round(left_encoder_ticks_float)),
                    right_ticks=int(round(right_encoder_ticks_float)),
                    gyro_z_radps=float(yaw_delta / self.dt + gyro_bias + sensor_rng.normal(0.0, gyro_noise_std)),
                )

                pose = next_pose
                final_step = step + 1
                poses.append(PoseSample(t=round(final_step * self.dt, 6), x=pose.x, y=pose.y, yaw=pose.yaw))
                world.reset_robot_pose(pose)
                self._append_odometry_sensor_samples(
                    encoder_samples=encoder_samples,
                    gyro_samples=gyro_samples,
                    t=final_step * self.dt,
                    frame_index=final_step,
                    sensor_state=sensor_state,
                )
                if state_estimation_mode and final_step % tag_update_stride == 0:
                    pending_april_tag_measurements.extend(
                        self._capture_april_tag_measurements(
                            pose=pose,
                            capture_step=final_step,
                            delivery_step=final_step + tag_latency_steps,
                            scenario=scenario,
                            rng=vision_rng,
                        )
                    )

                if final_step % render_stride == 0:
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=final_step * self.dt,
                        frame_index=final_step,
                        render_error=render_error,
                    )

                collision = self._first_collision(t=final_step * self.dt, pose=pose, scenario=scenario)
                if collision is not None:
                    collisions.append(collision)
                    status = "collision"
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=final_step * self.dt,
                        frame_index=final_step,
                        render_error=render_error,
                    )
                    break
        finally:
            world.close()

        elapsed = final_step * self.dt
        metrics = self._score_odometry(
            status=status,
            elapsed=elapsed,
            collisions=len(collisions),
            estimates=odometry_estimates,
            path_length=path_length,
            yaw_excitation=yaw_excitation,
            energy_used=energy_used,
            smoothness_cost=smoothness_cost,
        )
        replay = ReplayTrace(
            poses=poses,
            target=scenario.target,
            obstacles=scenario.obstacles,
            collisions=collisions,
            controls=controls,
            render_frames=render_frames,
            encoder_samples=encoder_samples,
            gyro_samples=gyro_samples,
            odometry_estimates=odometry_estimates,
            april_tag_pose_samples=april_tag_pose_samples,
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "challenge_mode": self.challenge.id,
                "mujoco_backend": world.backend_name,
                "simulation_mode": (
                    "kinematic_state_estimation_with_mujoco_render"
                    if state_estimation_mode and world.available
                    else "kinematic_odometry_with_mujoco_render"
                    if world.available
                    else "kinematic_fallback"
                ),
                "mujoco_timestep_sec": world.timestep if world.available else None,
                "mujoco_render_frames": len(render_frames),
                "mujoco_render_cameras": list(world.render_cameras),
                "mujoco_render_interval_sec": self.render_interval_sec if world.available else None,
                "mujoco_render_resolution": {
                    "width": world.render_width,
                    "height": world.render_height,
                },
                "mujoco_render_error": render_error,
                "robot_model": robot.model,
                "robot_dimensions_m": {
                    "length": robot.footprint_length_m,
                    "width": robot.footprint_width_m,
                    "height": robot.height_m,
                    "wheel_radius": robot.wheel_radius_m,
                    "wheel_base": robot.wheel_base_m,
                },
                "route_waypoints": [waypoint.model_dump() for waypoint in scenario.route],
                "ticks_per_rev": ticks_per_rev,
                "encoder_sample_count": len(encoder_samples),
                "gyro_sample_count": len(gyro_samples),
                "odometry_estimate_count": len(odometry_estimates),
                "april_tag_pose_sample_count": len(april_tag_pose_samples),
                "april_tag_layout": [tag.__dict__ for tag in scenario.april_tags],
                "trajectory_profile": self.challenge.defaults.get("trajectory_profile"),
                "april_tag_update_interval_sec": (
                    tag_update_stride * self.dt if state_estimation_mode else None
                ),
                "april_tag_latency_sec": tag_latency_steps * self.dt if state_estimation_mode else None,
                "april_tag_max_range_m": self.challenge.defaults.get("april_tag_max_range_m"),
                "april_tag_fov_deg": math.degrees(APRIL_TAG_CAMERA_FOV_RAD) if state_estimation_mode else None,
                "gyro_bias_radps": round(gyro_bias, 8),
                "gyro_noise_std_radps": gyro_noise_std,
                "wheel_slip_scales": {
                    "left": round(left_slip_scale, 8),
                    "right": round(right_slip_scale, 8),
                },
                "arena_bounds_m": scenario.bounds.__dict__,
            },
        )
        artifact, rerun_error = write_rerun_recording(
            challenge=self.challenge,
            seed=seed,
            replay=replay,
            metrics=metrics,
        )
        if artifact is not None:
            replay.artifacts.append(artifact)
        replay.metadata["rerun_artifacts"] = len(replay.artifacts)
        replay.metadata["rerun_export_error"] = rerun_error
        return RunResult(challenge_id=self.challenge.id, seed=seed, metrics=metrics, replay=replay)

    def _odometry_scenario(self) -> Scenario:
        start = Pose2D(x=0.0, y=0.0, yaw=0.0)
        target = Target(x=3.0, y=0.0, radius=float(self.challenge.success_conditions["final_position_error_m"]))
        route = [
            start,
            Pose2D(x=0.9, y=0.0, yaw=0.0),
            Pose2D(x=1.7, y=0.45, yaw=0.55),
            Pose2D(x=2.55, y=0.15, yaw=-0.15),
            Pose2D(x=3.2, y=0.65, yaw=0.55),
        ]
        return Scenario(start=start, target=target, route=route, obstacles=[], bounds=self.bounds)

    def _state_estimation_scenario(self) -> Scenario:
        start = Pose2D(x=0.0, y=0.0, yaw=0.0)
        target = Target(x=3.35, y=0.38, radius=float(self.challenge.success_conditions["final_position_error_m"]))
        route = [
            start,
            Pose2D(x=0.80, y=0.00, yaw=0.0),
            Pose2D(x=1.55, y=0.42, yaw=0.48),
            Pose2D(x=2.35, y=-0.08, yaw=-0.20),
            Pose2D(x=3.35, y=0.38, yaw=0.42),
        ]
        tags = (
            AprilTagLandmark(tag_id=1, x=0.65, y=1.28, yaw=-math.pi / 2),
            AprilTagLandmark(tag_id=2, x=1.45, y=-1.22, yaw=math.pi / 2),
            AprilTagLandmark(tag_id=3, x=2.25, y=1.30, yaw=-math.pi / 2),
            AprilTagLandmark(tag_id=4, x=3.18, y=-1.18, yaw=math.pi / 2),
            AprilTagLandmark(tag_id=5, x=3.85, y=0.92, yaw=-2.35),
        )
        return Scenario(start=start, target=target, route=route, obstacles=[], bounds=self.bounds, april_tags=tags)

    def _capture_april_tag_measurements(
        self,
        *,
        pose: Pose2D,
        capture_step: int,
        delivery_step: int,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> list[PendingAprilTagMeasurement]:
        if not scenario.april_tags:
            return []

        max_range = float(self.challenge.defaults.get("april_tag_max_range_m", 3.4))
        view_angle_limit = math.radians(float(self.challenge.defaults.get("april_tag_view_angle_limit_deg", 78.0)))
        position_base_std = float(self.challenge.defaults.get("april_tag_position_base_std_m", 0.012))
        position_distance_std = float(self.challenge.defaults.get("april_tag_position_distance_quadratic_std_m", 0.006))
        position_view_std = float(self.challenge.defaults.get("april_tag_position_view_angle_std_m", 0.020))
        yaw_base_std = float(self.challenge.defaults.get("april_tag_yaw_base_std_rad", math.radians(1.5)))
        yaw_distance_std = float(self.challenge.defaults.get("april_tag_yaw_distance_quadratic_std_rad", math.radians(0.55)))
        yaw_view_std = float(self.challenge.defaults.get("april_tag_yaw_view_angle_std_rad", math.radians(2.5)))
        capture_t = capture_step * self.dt
        delivery_t = delivery_step * self.dt
        latency = max(0.0, delivery_t - capture_t)

        pending: list[PendingAprilTagMeasurement] = []
        for tag in scenario.april_tags:
            dx = tag.x - pose.x
            dy = tag.y - pose.y
            distance = math.hypot(dx, dy)
            if distance > max_range:
                continue

            bearing = _wrap_angle(math.atan2(dy, dx) - pose.yaw)
            bearing_ratio = abs(bearing) / (APRIL_TAG_CAMERA_FOV_RAD / 2)
            if bearing_ratio > 1.0:
                continue

            tag_to_robot = math.atan2(pose.y - tag.y, pose.x - tag.x)
            view_angle = abs(_wrap_angle(tag_to_robot - tag.yaw))
            if view_angle > view_angle_limit:
                continue

            view_ratio = view_angle / view_angle_limit
            detection_probability = _clamp01(0.97 - 0.18 * (distance / max_range) ** 2 - 0.16 * bearing_ratio**2 - 0.10 * view_ratio**2)
            if rng.random() > detection_probability:
                continue

            position_std = (
                position_base_std
                + position_distance_std * distance * distance
                + position_view_std * view_ratio**2
                + 0.010 * bearing_ratio**2
            )
            yaw_std = yaw_base_std + yaw_distance_std * distance * distance + yaw_view_std * view_ratio**2
            ambiguity = _clamp01(
                0.025
                + 0.080 * (distance / max_range) ** 2
                + 0.110 * bearing_ratio**2
                + 0.070 * view_ratio**2
                + float(rng.normal(0.0, 0.012))
            )
            measured_x = pose.x + float(rng.normal(0.0, position_std))
            measured_y = pose.y + float(rng.normal(0.0, position_std * (1.0 + 0.35 * abs(math.sin(bearing)))))
            measured_yaw = _wrap_angle(pose.yaw + float(rng.normal(0.0, yaw_std)))
            measured_pose = ApiPose2d(x=measured_x, y=measured_y, yaw=measured_yaw)
            measurement = ApiAprilTagPoseEstimate(
                tag_id=tag.tag_id,
                timestamp=round(capture_t, 6),
                pose=measured_pose,
                distance_m=round(distance, 6),
                bearing_rad=round(bearing, 6),
                position_std_m=round(position_std, 6),
                yaw_std_rad=round(yaw_std, 6),
                ambiguity=round(ambiguity, 6),
            )
            sample = AprilTagPoseSample(
                t=round(delivery_t, 6),
                capture_t=round(capture_t, 6),
                frame_index=delivery_step,
                tag_id=tag.tag_id,
                x=round(measured_x, 6),
                y=round(measured_y, 6),
                yaw=round(measured_yaw, 6),
                distance_m=round(distance, 6),
                bearing_rad=round(bearing, 6),
                position_std_m=round(position_std, 6),
                yaw_std_rad=round(yaw_std, 6),
                latency_sec=round(latency, 6),
                ambiguity=round(ambiguity, 6),
            )
            pending.append(
                PendingAprilTagMeasurement(
                    delivery_step=delivery_step,
                    measurement=measurement,
                    sample=sample,
                )
            )

        return pending

    def _is_state_estimation_challenge(self) -> bool:
        return self.challenge.defaults.get("runner") == DIFF_DRIVE_STATE_ESTIMATION_RUNNER

    def _odometry_trajectory_command(self, *, t: float, state_estimation_mode: bool) -> tuple[float, float]:
        if state_estimation_mode:
            forward = 0.22
            angular = 0.35 * math.sin(1.0 * t)
        else:
            forward = 0.32
            angular = 0.65 * math.sin(1.1 * t)

        left = forward - 0.5 * self.challenge.robot.wheel_base_m * angular
        right = forward + 0.5 * self.challenge.robot.wheel_base_m * angular
        max_velocity = self.challenge.robot.max_wheel_velocity_mps
        return (
            max(-max_velocity, min(max_velocity, left)),
            max(-max_velocity, min(max_velocity, right)),
        )

    def _call_odometry_policy_step(
        self,
        policy: RobotPolicyProtocol,
        robot: DifferentialDriveOdometryApi,
    ) -> ApiPose2d:
        try:
            robot._clear_step_outputs()
            step_result = policy.step(robot)
            if step_result is not None:
                raise SimulationError(
                    "RobotPolicy.step(robot) should call robot.submit_odometry(...), not return an action."
                )
            return robot._consume_odometry_estimate()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _append_odometry_sensor_samples(
        self,
        *,
        encoder_samples: list[EncoderTraceSample],
        gyro_samples: list[GyroTraceSample],
        t: float,
        frame_index: int,
        sensor_state: OdometrySensorState,
    ) -> None:
        if not any(sample.frame_index == frame_index for sample in encoder_samples):
            encoder_samples.append(
                EncoderTraceSample(
                    t=round(t, 6),
                    frame_index=frame_index,
                    left_ticks=sensor_state.left_ticks,
                    right_ticks=sensor_state.right_ticks,
                )
            )
        if not any(sample.frame_index == frame_index for sample in gyro_samples):
            gyro_samples.append(
                GyroTraceSample(
                    t=round(t, 6),
                    frame_index=frame_index,
                    yaw_rate_radps=round(sensor_state.gyro_z_radps, 8),
                )
            )

    def _score_odometry(
        self,
        *,
        status: str,
        elapsed: float,
        collisions: int,
        estimates: list[OdometryEstimateSample],
        path_length: float,
        yaw_excitation: float,
        energy_used: float,
        smoothness_cost: float,
    ) -> ScoreMetrics:
        position_errors = [sample.position_error_m for sample in estimates]
        yaw_errors = [sample.yaw_error_rad for sample in estimates]
        final_position_error = position_errors[-1] if position_errors else float("inf")
        mean_position_error = float(np.mean(position_errors)) if position_errors else float("inf")
        max_position_error = max(position_errors) if position_errors else float("inf")
        final_yaw_error = yaw_errors[-1] if yaw_errors else float("inf")
        mean_yaw_error = float(np.mean(yaw_errors)) if yaw_errors else float("inf")

        final_position_tolerance = float(self.challenge.success_conditions["final_position_error_m"])
        mean_position_tolerance = float(self.challenge.success_conditions["mean_position_error_m"])
        final_yaw_tolerance = float(self.challenge.success_conditions["final_yaw_error_rad"])

        success = (
            status != "collision"
            and collisions == 0
            and final_position_error <= final_position_tolerance
            and mean_position_error <= mean_position_tolerance
            and final_yaw_error <= final_yaw_tolerance
        )
        final_status = "success" if success else status

        final_position_points = 35.0 * _clamp01(1.0 - final_position_error / (final_position_tolerance * 3.0))
        mean_position_points = 35.0 * _clamp01(1.0 - mean_position_error / (mean_position_tolerance * 3.0))
        yaw_points = 25.0 * _clamp01(1.0 - final_yaw_error / (final_yaw_tolerance * 3.0))
        safety_points = max(0.0, 5.0 - 5.0 * collisions)
        total = min(
            100.0,
            final_position_points
            + mean_position_points
            + yaw_points
            + safety_points
        )

        return ScoreMetrics(
            metric_kind="odometry",
            score=round(total, 3),
            success=success,
            status=final_status,  # type: ignore[arg-type]
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(final_position_error, 6),
            heading_error_rad=round(final_yaw_error, 6),
            collision_count=collisions,
            path_length_m=round(path_length, 6),
            energy_used=round(energy_used, 6),
            smoothness_cost=round(smoothness_cost, 6),
            final_position_error_m=round(final_position_error, 6),
            mean_position_error_m=round(mean_position_error, 6),
            max_position_error_m=round(max_position_error, 6),
            final_yaw_error_rad=round(final_yaw_error, 6),
            mean_yaw_error_rad=round(mean_yaw_error, 6),
            excitation_distance_m=round(path_length, 6),
            excitation_yaw_rad=round(yaw_excitation, 6),
        )


class DifferentialDriveSlamRunner(DifferentialDriveOdometryRunner):
    """Differential-drive runner that grades 2D lidar SLAM pose and map submissions."""

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        max_step_count = max_steps or self.default_max_steps
        scenario = self._slam_scenario(seed)
        robot = self.challenge.robot
        lidar = robot.lidars[0]
        ticks_per_rev = int(self.challenge.defaults.get("ticks_per_rev", 392))
        max_map_points = int(self.challenge.defaults.get("max_map_points", 1600))
        angle_min, angle_increment = self._lidar_angle_model(ray_count=lidar.num_rays, fov_deg=lidar.fov_deg)
        sensor_rng = np.random.default_rng(seed + 91_003)
        parameter_rng = random.Random(seed + 17_123)
        encoder_scale_range = float(self.challenge.defaults.get("encoder_scale_error_range", 0.0))
        left_encoder_scale = 1.0 + parameter_rng.uniform(-encoder_scale_range, encoder_scale_range)
        right_encoder_scale = 1.0 + parameter_rng.uniform(-encoder_scale_range, encoder_scale_range)
        gyro_bias = parameter_rng.uniform(
            -float(self.challenge.defaults.get("gyro_bias_range_radps", 0.0)),
            float(self.challenge.defaults.get("gyro_bias_range_radps", 0.0)),
        )
        gyro_noise_std = float(self.challenge.defaults.get("gyro_noise_std_radps", 0.0))

        world = MujocoWorld.create(
            target=scenario.target,
            obstacles=scenario.obstacles,
            bounds=scenario.bounds.__dict__,
            robot=robot,
            target_kind="target_disk",
        )

        poses: list[PoseSample] = []
        controls: list[ControlSample] = []
        collisions: list[CollisionEvent] = []
        render_frames: list[RenderFrame] = []
        lidar_scans: list[LidarScanSample] = []
        encoder_samples: list[EncoderTraceSample] = []
        gyro_samples: list[GyroTraceSample] = []
        slam_estimates: list[OdometryEstimateSample] = []
        final_map_points: tuple[ApiMapPoint2d, ...] = ()
        render_error: str | None = None
        rerun_error: str | None = None
        path_length = 0.0
        yaw_excitation = 0.0
        energy_used = 0.0
        smoothness_cost = 0.0
        last_control: tuple[float, float] | None = None
        left_encoder_ticks_float = 0.0
        right_encoder_ticks_float = 0.0
        sensor_state = OdometrySensorState(left_ticks=0, right_ticks=0, gyro_z_radps=0.0)
        status = "timeout"
        final_step = 0

        policy_robot = DifferentialDriveSlamApi(
            wheel_base_m=robot.wheel_base_m,
            wheel_radius_m=robot.wheel_radius_m,
            max_wheel_velocity_mps=robot.max_wheel_velocity_mps,
            ticks_per_rev=ticks_per_rev,
            dt=self.dt,
            max_steps=max_step_count,
            seed=seed,
            lidar_angle_min_rad=angle_min,
            lidar_angle_increment_rad=angle_increment,
            lidar_max_range_m=lidar.max_range_m,
            max_map_points=max_map_points,
        )

        try:
            render_stride = max(1, round(self.render_interval_sec / self.dt))
            for step in range(max_step_count + 1):
                t = step * self.dt
                pose = self._slam_pose_at(step=step, max_steps=max_step_count)
                poses.append(PoseSample(t=round(t, 6), x=pose.x, y=pose.y, yaw=pose.yaw))
                world.reset_robot_pose(pose)

                lidar_scan = self._noisy_lidar_scan(
                    t=t,
                    frame_index=step,
                    pose=pose,
                    scenario=scenario,
                    rng=sensor_rng,
                )
                policy_robot._update(
                    time=round(t, 6),
                    encoder_values=(sensor_state.left_ticks, sensor_state.right_ticks),
                    gyro_z=sensor_state.gyro_z_radps,
                    collision_count=len(collisions),
                    lidar_ranges=np.asarray(lidar_scan.ranges_m, dtype=float),
                )
                slam_pose, map_points = self._call_slam_policy_step(policy, policy_robot)
                final_map_points = map_points
                position_error = math.hypot(slam_pose.x - pose.x, slam_pose.y - pose.y)
                yaw_error = abs(_wrap_angle(slam_pose.yaw - pose.yaw))
                slam_estimates.append(
                    OdometryEstimateSample(
                        t=round(t, 6),
                        frame_index=step,
                        x=round(slam_pose.x, 6),
                        y=round(slam_pose.y, 6),
                        yaw=round(slam_pose.yaw, 6),
                        position_error_m=round(position_error, 6),
                        yaw_error_rad=round(yaw_error, 6),
                    )
                )

                self._append_odometry_sensor_samples(
                    encoder_samples=encoder_samples,
                    gyro_samples=gyro_samples,
                    t=t,
                    frame_index=step,
                    sensor_state=sensor_state,
                )
                if step % render_stride == 0 or step == max_step_count:
                    if not any(scan.frame_index == step for scan in lidar_scans):
                        lidar_scans.append(lidar_scan)
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t,
                        frame_index=step,
                        render_error=render_error,
                    )

                collision = self._first_collision(t=t, pose=pose, scenario=scenario)
                if collision is not None:
                    collisions.append(collision)
                    status = "collision"
                    final_step = step
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t,
                        frame_index=step,
                        render_error=render_error,
                    )
                    break

                if step >= max_step_count:
                    final_step = step
                    break

                next_pose = self._slam_pose_at(step=step + 1, max_steps=max_step_count)
                left_distance, right_distance, yaw_delta = self._wheel_deltas_between(pose=pose, next_pose=next_pose)
                left_velocity = left_distance / self.dt
                right_velocity = right_distance / self.dt
                controls.append(
                    ControlSample(
                        t=round(t, 6),
                        left_wheel_velocity=round(left_velocity, 6),
                        right_wheel_velocity=round(right_velocity, 6),
                    )
                )
                if last_control is not None:
                    smoothness_cost += (
                        (left_velocity - last_control[0]) ** 2 + (right_velocity - last_control[1]) ** 2
                    ) * self.dt
                last_control = (left_velocity, right_velocity)
                energy_used += (left_velocity**2 + right_velocity**2) * self.dt
                path_length += math.hypot(next_pose.x - pose.x, next_pose.y - pose.y)
                yaw_excitation += abs(yaw_delta)

                left_encoder_ticks_float += _wheel_velocity_to_ticks(
                    velocity=left_velocity * left_encoder_scale,
                    dt=self.dt,
                    wheel_radius=robot.wheel_radius_m,
                    ticks_per_rev=ticks_per_rev,
                )
                right_encoder_ticks_float += _wheel_velocity_to_ticks(
                    velocity=right_velocity * right_encoder_scale,
                    dt=self.dt,
                    wheel_radius=robot.wheel_radius_m,
                    ticks_per_rev=ticks_per_rev,
                )
                sensor_state = OdometrySensorState(
                    left_ticks=int(round(left_encoder_ticks_float)),
                    right_ticks=int(round(right_encoder_ticks_float)),
                    gyro_z_radps=float(yaw_delta / self.dt + gyro_bias + sensor_rng.normal(0.0, gyro_noise_std)),
                )
        finally:
            world.close()

        elapsed = final_step * self.dt
        truth_map_points = self._visible_slam_map(scenario=scenario, max_steps=max_step_count)
        submitted_points = _downsample_map_points(
            [(point.x, point.y) for point in final_map_points],
            grid_size=float(self.challenge.defaults.get("submitted_map_grid_m", 0.07)),
            limit=max_map_points,
        )
        metrics = self._score_slam(
            status=status,
            elapsed=elapsed,
            collisions=len(collisions),
            estimates=slam_estimates,
            submitted_points=submitted_points,
            truth_points=truth_map_points,
            path_length=path_length,
            yaw_excitation=yaw_excitation,
            energy_used=energy_used,
            smoothness_cost=smoothness_cost,
        )
        replay = ReplayTrace(
            poses=poses,
            target=scenario.target,
            obstacles=scenario.obstacles,
            collisions=collisions,
            controls=controls,
            render_frames=render_frames,
            lidar_scans=lidar_scans,
            encoder_samples=encoder_samples,
            gyro_samples=gyro_samples,
            odometry_estimates=slam_estimates,
            slam_map_points=[MapPoint2D(x=round(x, 6), y=round(y, 6)) for x, y in submitted_points],
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "challenge_mode": self.challenge.id,
                "mujoco_backend": world.backend_name,
                "simulation_mode": "kinematic_slam_with_mujoco_render" if world.available else "kinematic_fallback",
                "mujoco_timestep_sec": world.timestep if world.available else None,
                "mujoco_render_frames": len(render_frames),
                "mujoco_render_cameras": list(world.render_cameras),
                "mujoco_render_interval_sec": self.render_interval_sec if world.available else None,
                "mujoco_render_resolution": {
                    "width": world.render_width,
                    "height": world.render_height,
                },
                "mujoco_render_error": render_error,
                "robot_model": robot.model,
                "robot_dimensions_m": {
                    "length": robot.footprint_length_m,
                    "width": robot.footprint_width_m,
                    "height": robot.height_m,
                    "wheel_radius": robot.wheel_radius_m,
                    "wheel_base": robot.wheel_base_m,
                },
                "ticks_per_rev": ticks_per_rev,
                "lidar_angle_min_rad": angle_min,
                "lidar_angle_increment_rad": angle_increment,
                "lidar_max_range_m": lidar.max_range_m,
                "lidar_noise_std_m": self.challenge.defaults.get("lidar_noise_std_m"),
                "encoder_scale_error": {
                    "left": round(left_encoder_scale, 8),
                    "right": round(right_encoder_scale, 8),
                },
                "gyro_bias_radps": round(gyro_bias, 8),
                "gyro_noise_std_radps": gyro_noise_std,
                "encoder_sample_count": len(encoder_samples),
                "gyro_sample_count": len(gyro_samples),
                "lidar_scan_count": len(lidar_scans),
                "slam_estimate_count": len(slam_estimates),
                "slam_map_point_count": len(submitted_points),
                "ground_truth_visible_map_point_count": len(truth_map_points),
                "trajectory_profile": self.challenge.defaults.get("trajectory_profile"),
                "loop_closure": "start_pose_revisited_at_final_step",
                "route_waypoints": [waypoint.model_dump() for waypoint in scenario.route],
                "arena_bounds_m": scenario.bounds.__dict__,
            },
        )
        artifact, rerun_error = write_rerun_recording(
            challenge=self.challenge,
            seed=seed,
            replay=replay,
            metrics=metrics,
        )
        if artifact is not None:
            replay.artifacts.append(artifact)
        replay.metadata["rerun_artifacts"] = len(replay.artifacts)
        replay.metadata["rerun_export_error"] = rerun_error
        return RunResult(challenge_id=self.challenge.id, seed=seed, metrics=metrics, replay=replay)

    def _slam_scenario(self, seed: int) -> Scenario:
        rng = random.Random(seed)
        target = Target(
            x=0.0,
            y=0.0,
            radius=float(self.challenge.success_conditions["final_pose_error_m"]),
        )
        jitter = float(self.challenge.defaults.get("landmark_jitter_m", 0.035))
        obstacles = [
            Obstacle(id="north_east_reflector", x=2.25 + rng.uniform(-jitter, jitter), y=1.15 + rng.uniform(-jitter, jitter), radius=0.18),
            Obstacle(id="south_west_reflector", x=-2.25 + rng.uniform(-jitter, jitter), y=-1.15 + rng.uniform(-jitter, jitter), radius=0.18),
            Obstacle(id="east_column", x=2.45 + rng.uniform(-jitter, jitter), y=-0.75 + rng.uniform(-jitter, jitter), radius=0.16),
            Obstacle(id="west_column", x=-2.45 + rng.uniform(-jitter, jitter), y=0.75 + rng.uniform(-jitter, jitter), radius=0.16),
            Obstacle(id="north_box", x=0.0 + rng.uniform(-jitter, jitter), y=2.00 + rng.uniform(-jitter, jitter), radius=0.20),
            Obstacle(id="south_box", x=0.0 + rng.uniform(-jitter, jitter), y=-2.00 + rng.uniform(-jitter, jitter), radius=0.20),
        ]
        route = [
            Pose2D(x=0.0, y=0.0, yaw=0.0),
            Pose2D(x=1.35, y=-1.05, yaw=-0.45),
            Pose2D(x=1.80, y=0.95, yaw=1.80),
            Pose2D(x=0.0, y=0.0, yaw=0.0),
            Pose2D(x=-1.80, y=-0.95, yaw=-1.35),
            Pose2D(x=-1.35, y=1.05, yaw=0.45),
            Pose2D(x=0.0, y=0.0, yaw=0.0),
        ]
        return Scenario(start=route[0], target=target, route=route, obstacles=obstacles, bounds=self.bounds)

    def _slam_pose_at(self, *, step: int, max_steps: int) -> Pose2D:
        progress = 0.0 if max_steps <= 0 else step / max_steps
        amplitude_x = float(self.challenge.defaults.get("trajectory_amplitude_x_m", 2.05))
        amplitude_y = float(self.challenge.defaults.get("trajectory_amplitude_y_m", 0.95))
        x_raw = amplitude_x * math.sin(2.0 * math.pi * progress)
        y_raw = amplitude_y * math.sin(4.0 * math.pi * progress)
        dx_raw = amplitude_x * 2.0 * math.pi * math.cos(2.0 * math.pi * progress)
        dy_raw = amplitude_y * 4.0 * math.pi * math.cos(4.0 * math.pi * progress)
        initial_yaw = math.atan2(amplitude_y * 4.0 * math.pi, amplitude_x * 2.0 * math.pi)
        cos_initial = math.cos(-initial_yaw)
        sin_initial = math.sin(-initial_yaw)
        x = cos_initial * x_raw - sin_initial * y_raw
        y = sin_initial * x_raw + cos_initial * y_raw
        yaw = _wrap_angle(math.atan2(dy_raw, dx_raw) - initial_yaw)
        return Pose2D(x=x, y=y, yaw=yaw)

    def _lidar_angle_model(self, *, ray_count: int, fov_deg: float) -> tuple[float, float]:
        fov_rad = math.radians(fov_deg)
        if ray_count <= 1:
            return 0.0, 0.0
        return -fov_rad / 2.0, fov_rad / ray_count

    def _noisy_lidar_scan(
        self,
        *,
        t: float,
        frame_index: int,
        pose: Pose2D,
        scenario: Scenario,
        rng: np.random.Generator,
    ) -> LidarScanSample:
        base_scan = self._lidar_scan(t=t, frame_index=frame_index, pose=pose, scenario=scenario)
        noise_std = float(self.challenge.defaults.get("lidar_noise_std_m", 0.0))
        quantization = float(self.challenge.defaults.get("lidar_quantization_m", 0.0))
        noisy_ranges: list[float] = []
        for value in base_scan.ranges_m:
            measured = float(value)
            if measured < base_scan.max_range_m and noise_std > 0.0:
                measured += float(rng.normal(0.0, noise_std))
            measured = max(0.03, min(base_scan.max_range_m, measured))
            if quantization > 0.0:
                measured = round(measured / quantization) * quantization
            noisy_ranges.append(round(measured, 4))

        return LidarScanSample(
            t=base_scan.t,
            frame_index=base_scan.frame_index,
            frame=base_scan.frame,
            angles_rad=base_scan.angles_rad,
            ranges_m=noisy_ranges,
            max_range_m=base_scan.max_range_m,
        )

    def _wheel_deltas_between(self, *, pose: Pose2D, next_pose: Pose2D) -> tuple[float, float, float]:
        dx = next_pose.x - pose.x
        dy = next_pose.y - pose.y
        forward_distance = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
        yaw_delta = _wrap_angle(next_pose.yaw - pose.yaw)
        half_track_yaw = 0.5 * self.challenge.robot.wheel_base_m * yaw_delta
        return forward_distance - half_track_yaw, forward_distance + half_track_yaw, yaw_delta

    def _call_slam_policy_step(
        self,
        policy: RobotPolicyProtocol,
        robot: DifferentialDriveSlamApi,
    ) -> tuple[ApiPose2d, tuple[ApiMapPoint2d, ...]]:
        try:
            robot._clear_step_outputs()
            step_result = policy.step(robot)
            if step_result is not None:
                raise SimulationError(
                    "RobotPolicy.step(robot) should call robot.submit_slam(...), not return an action."
                )
            return robot._consume_slam_submission()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _visible_slam_map(self, *, scenario: Scenario, max_steps: int) -> list[tuple[float, float]]:
        raw_points: list[tuple[float, float]] = []
        stride = max(1, int(self.challenge.defaults.get("ground_truth_map_stride_steps", 4)))
        for step in range(0, max_steps + 1, stride):
            pose = self._slam_pose_at(step=step, max_steps=max_steps)
            scan = self._lidar_scan(t=step * self.dt, frame_index=step, pose=pose, scenario=scenario)
            for angle, distance in zip(scan.angles_rad, scan.ranges_m, strict=False):
                if distance >= scan.max_range_m - 0.02:
                    continue
                raw_points.append(
                    (
                        pose.x + distance * math.cos(pose.yaw + angle),
                        pose.y + distance * math.sin(pose.yaw + angle),
                    )
                )
        return _downsample_map_points(
            raw_points,
            grid_size=float(self.challenge.defaults.get("ground_truth_map_grid_m", 0.08)),
            limit=int(self.challenge.defaults.get("ground_truth_max_map_points", 2400)),
        )

    def _score_slam(
        self,
        *,
        status: str,
        elapsed: float,
        collisions: int,
        estimates: list[OdometryEstimateSample],
        submitted_points: list[tuple[float, float]],
        truth_points: list[tuple[float, float]],
        path_length: float,
        yaw_excitation: float,
        energy_used: float,
        smoothness_cost: float,
    ) -> ScoreMetrics:
        final_position_error = estimates[-1].position_error_m if estimates else float("inf")
        mean_position_error = float(np.mean([sample.position_error_m for sample in estimates])) if estimates else float("inf")
        max_position_error = max((sample.position_error_m for sample in estimates), default=float("inf"))
        final_yaw_error = estimates[-1].yaw_error_rad if estimates else float("inf")
        mean_yaw_error = float(np.mean([sample.yaw_error_rad for sample in estimates])) if estimates else float("inf")
        submitted_array = np.asarray(submitted_points, dtype=float)
        truth_array = np.asarray(truth_points, dtype=float)
        submitted_to_truth = _nearest_distances(source=submitted_array, target=truth_array)
        truth_to_submitted = _nearest_distances(source=truth_array, target=submitted_array)

        map_chamfer = float("inf")
        map_coverage = 0.0
        false_positive_ratio = 1.0
        if len(submitted_to_truth) > 0 and len(truth_to_submitted) > 0:
            map_chamfer = 0.5 * (float(np.mean(submitted_to_truth)) + float(np.mean(truth_to_submitted)))
            coverage_radius = float(self.challenge.success_conditions["map_coverage_radius_m"])
            false_positive_radius = float(self.challenge.success_conditions["map_false_positive_radius_m"])
            map_coverage = float(np.mean(truth_to_submitted <= coverage_radius))
            false_positive_ratio = float(np.mean(submitted_to_truth > false_positive_radius))

        final_position_tolerance = float(self.challenge.success_conditions["final_pose_error_m"])
        final_yaw_tolerance = float(self.challenge.success_conditions["final_yaw_error_rad"])
        map_chamfer_tolerance = float(self.challenge.success_conditions["map_chamfer_error_m"])
        map_coverage_tolerance = float(self.challenge.success_conditions["map_coverage_ratio"])
        false_positive_tolerance = float(self.challenge.success_conditions["map_false_positive_ratio"])
        success = (
            status != "collision"
            and collisions == 0
            and final_position_error <= final_position_tolerance
            and final_yaw_error <= final_yaw_tolerance
            and map_chamfer <= map_chamfer_tolerance
            and map_coverage >= map_coverage_tolerance
            and false_positive_ratio <= false_positive_tolerance
        )
        final_status = "success" if success else ("collision" if collisions else "accuracy_error")

        final_pose_points = 20.0 * _clamp01(1.0 - final_position_error / (final_position_tolerance * 3.0))
        yaw_points = 10.0 * _clamp01(1.0 - final_yaw_error / (final_yaw_tolerance * 3.0))
        map_accuracy_points = 35.0 * _clamp01(1.0 - map_chamfer / (map_chamfer_tolerance * 3.0))
        coverage_points = 25.0 * _clamp01(map_coverage / max(map_coverage_tolerance, 1e-9))
        precision_points = 10.0 * _clamp01(1.0 - false_positive_ratio / max(false_positive_tolerance * 2.5, 1e-9))
        safety_points = 0.0 if collisions else 5.0
        total = min(
            100.0,
            final_pose_points + yaw_points + map_accuracy_points + coverage_points + precision_points + safety_points,
        )

        return ScoreMetrics(
            metric_kind="slam",
            score=round(total, 3),
            success=success,
            status=final_status,  # type: ignore[arg-type]
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(final_position_error, 6),
            heading_error_rad=round(final_yaw_error, 6),
            collision_count=collisions,
            path_length_m=round(path_length, 6),
            energy_used=round(energy_used, 6),
            smoothness_cost=round(smoothness_cost, 6),
            final_position_error_m=round(final_position_error, 6),
            mean_position_error_m=round(mean_position_error, 6),
            max_position_error_m=round(max_position_error, 6),
            final_yaw_error_rad=round(final_yaw_error, 6),
            mean_yaw_error_rad=round(mean_yaw_error, 6),
            excitation_distance_m=round(path_length, 6),
            excitation_yaw_rad=round(yaw_excitation, 6),
            map_chamfer_error_m=round(map_chamfer, 6),
            map_coverage_ratio=round(map_coverage, 6),
            map_false_positive_ratio=round(false_positive_ratio, 6),
            map_point_count=len(submitted_points),
        )


def _relative_pose(*, pose: Pose2D, target: Target, target_yaw: float) -> tuple[float, float, float]:
    dx = target.x - pose.x
    dy = target.y - pose.y
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return (
        cos_yaw * dx + sin_yaw * dy,
        -sin_yaw * dx + cos_yaw * dy,
        _wrap_angle(target_yaw - pose.yaw),
    )


def _charger_camera_frame(*, rel_x: float, rel_y: float, visible: bool) -> np.ndarray:
    image = np.zeros((CHARGER_CAMERA_HEIGHT, CHARGER_CAMERA_WIDTH, 3), dtype=np.uint8)
    image[:, :, 0] = 28
    image[:, :, 1] = 32
    image[:, :, 2] = 38

    horizon = CHARGER_CAMERA_HEIGHT // 3
    image[:horizon, :, :] = np.array([42, 48, 58], dtype=np.uint8)
    for row in range(horizon, CHARGER_CAMERA_HEIGHT):
        shade = 52 + int(42 * (row - horizon) / max(1, CHARGER_CAMERA_HEIGHT - horizon))
        image[row, :, :] = np.array([shade, shade + 4, shade + 8], dtype=np.uint8)

    if not visible:
        return image

    distance = max(0.18, math.hypot(rel_x, rel_y))
    bearing = math.atan2(rel_y, max(rel_x, 1e-6))
    center = int(
        CHARGER_CAMERA_WIDTH / 2
        + (bearing / (CHARGER_CAMERA_FOV_RAD / 2)) * (CHARGER_CAMERA_WIDTH * 0.45)
    )
    marker_height = int(max(14, min(CHARGER_CAMERA_HEIGHT * 0.82, CHARGER_CAMERA_HEIGHT * 0.52 / distance)))
    marker_width = int(max(6, min(34, 16 / distance)))
    top = max(4, horizon - marker_height // 4)
    bottom = min(CHARGER_CAMERA_HEIGHT - 5, top + marker_height)
    left = max(0, center - marker_width // 2)
    right = min(CHARGER_CAMERA_WIDTH, center + marker_width // 2)

    image[top:bottom, left:right, :] = np.array([42, 210, 118], dtype=np.uint8)
    stripe_left = max(0, center - marker_width)
    stripe_right = min(CHARGER_CAMERA_WIDTH, center + marker_width)
    image[max(top - 3, 0) : top, stripe_left:stripe_right, :] = np.array([220, 238, 246], dtype=np.uint8)
    return image


def _wheel_velocity_to_ticks(*, velocity: float, dt: float, wheel_radius: float, ticks_per_rev: int) -> float:
    wheel_circumference = 2.0 * math.pi * wheel_radius
    if wheel_circumference <= 0.0:
        return 0.0
    return (velocity * dt / wheel_circumference) * ticks_per_rev


def _downsample_map_points(
    points: list[tuple[float, float]],
    *,
    grid_size: float,
    limit: int,
) -> list[tuple[float, float]]:
    if grid_size <= 0.0:
        grid_size = 0.05

    selected: dict[tuple[int, int], tuple[float, float]] = {}
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        key = (round(x / grid_size), round(y / grid_size))
        if key not in selected:
            selected[key] = (float(x), float(y))
        if len(selected) >= limit:
            break
    return list(selected.values())


def _nearest_distances(*, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.size == 0 or target.size == 0:
        return np.array([], dtype=float)

    source_2d = source.reshape((-1, 2))
    target_2d = target.reshape((-1, 2))
    distances: list[np.ndarray] = []
    for start in range(0, len(source_2d), 256):
        chunk = source_2d[start : start + 256]
        deltas = chunk[:, None, :] - target_2d[None, :, :]
        distances.append(np.sqrt(np.sum(deltas * deltas, axis=2)).min(axis=1))
    return np.concatenate(distances) if distances else np.array([], dtype=float)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _distance_to_target(pose: Pose2D, target: Target) -> float:
    return math.hypot(target.x - pose.x, target.y - pose.y)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
