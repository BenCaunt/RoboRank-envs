from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Pose2d:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Target2d:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class CircleObstacle:
    id: str
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class MapPoint2d:
    x: float
    y: float


@dataclass(frozen=True)
class Pose3d:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float
    vy: float
    vz: float


@dataclass(frozen=True)
class Gate3d:
    id: str
    x: float
    y: float
    z: float
    yaw: float
    width_m: float
    height_m: float


@dataclass(frozen=True)
class ImuSample:
    t: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass(frozen=True)
class CollisionDecision:
    contact: bool
    severity: str


@dataclass(frozen=True)
class IKTarget:
    index: int
    x: float
    y: float
    z: float
    tolerance_m: float
    label: str = "target"


@dataclass(frozen=True)
class AprilTagPoseEstimate:
    tag_id: int
    timestamp: float
    pose: Pose2d
    distance_m: float
    bearing_rad: float
    position_std_m: float
    yaw_std_rad: float
    ambiguity: float


@dataclass(frozen=True)
class CartPoleState:
    t: float
    cart_position_m: float
    cart_velocity_mps: float
    pole_angle_rad: float
    pole_angular_velocity_radps: float


@dataclass(frozen=True)
class MinimumPhaseState:
    output_m: float
    output_velocity_mps: float


@dataclass(frozen=True)
class MotionState1D:
    t: float
    position_m: float
    velocity_mps: float
    acceleration_mps2: float


@dataclass(frozen=True)
class MotionTarget1D:
    position_m: float
    velocity_mps: float = 0.0


class MobileRobot:
    """Base class for injected robot APIs."""


class Actuator:
    """Base class for injected actuator fixture APIs."""


class AerialRobot:
    """Base class for injected aerial robot APIs."""


class ControlSystem:
    """Base class for injected low-dimensional control systems."""


class KinematicMechanism:
    """Base class for injected kinematic mechanism APIs."""


class InverseKinematicsTask(KinematicMechanism):
    """Runtime API injected into inverse-kinematics policy code."""

    def __init__(
        self,
        *,
        mechanism: str,
        joint_count: int,
        link_lengths_m: tuple[float, ...] | list[float],
        joint_limits_rad: tuple[tuple[float, float], ...] | list[list[float]],
        base_height_m: float,
        base_spacing_m: float | None,
        tolerance_m: float,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.mechanism = str(mechanism)
        self.joint_count = int(joint_count)
        self.link_lengths_m = tuple(float(length) for length in link_lengths_m)
        self.joint_limits_rad = tuple((float(limit[0]), float(limit[1])) for limit in joint_limits_rad)
        self.base_height_m = float(base_height_m)
        self.base_spacing_m = None if base_spacing_m is None else float(base_spacing_m)
        self.tolerance_m = float(tolerance_m)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.time = 0.0
        self.target_index = 0
        self.target_count = 0
        self._target = IKTarget(index=0, x=0.0, y=0.0, z=0.0, tolerance_m=self.tolerance_m)
        self._joint_angles: tuple[float, ...] | None = None

    @property
    def link_1_m(self) -> float:
        return self.link_lengths_m[0] if self.link_lengths_m else 0.0

    @property
    def link_2_m(self) -> float:
        return self.link_lengths_m[1] if len(self.link_lengths_m) > 1 else 0.0

    def get_target(self) -> IKTarget:
        return self._target

    def joint_limits(self) -> tuple[tuple[float, float], ...]:
        return self.joint_limits_rad

    def progress(self) -> tuple[int, int]:
        return self.target_index, self.target_count

    def submit_joint_angles(self, angles: list[float] | tuple[float, ...]) -> None:
        if len(angles) != self.joint_count:
            raise ValueError(f"submit_joint_angles expects {self.joint_count} joint angles.")

        parsed: list[float] = []
        for angle in angles:
            try:
                value = float(angle)
            except (TypeError, ValueError) as exc:
                raise ValueError("Joint angles must be numeric radians.") from exc
            if not math.isfinite(value):
                raise ValueError("Joint angles must be finite radians.")
            parsed.append(value)
        self._joint_angles = tuple(parsed)

    def _update(self, *, time: float, target: IKTarget, target_count: int) -> None:
        self.time = float(time)
        self.target_index = int(target.index)
        self.target_count = int(target_count)
        self._target = target

    def _clear_submission(self) -> None:
        self._joint_angles = None

    def _consume_joint_angles(self) -> tuple[float, ...]:
        if self._joint_angles is None:
            raise ValueError("RobotPolicy.step(arm) must call arm.submit_joint_angles([...]).")
        return self._joint_angles


class CartPole(ControlSystem):
    """Runtime API injected into cart-pole policy code."""

    def __init__(
        self,
        *,
        cart_mass_kg: float,
        pole_mass_kg: float,
        pole_com_length_m: float,
        pole_length_m: float,
        track_half_width_m: float,
        max_force_n: float,
        gravity_mps2: float,
        minimum_phase_gain_m: float,
        target_position_m: float,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.cart_mass_kg = cart_mass_kg
        self.pole_mass_kg = pole_mass_kg
        self.pole_com_length_m = pole_com_length_m
        self.pole_length_m = pole_length_m
        self.track_half_width_m = track_half_width_m
        self.max_force_n = max_force_n
        self.gravity_mps2 = gravity_mps2
        self.minimum_phase_gain_m = minimum_phase_gain_m
        self.target_position_m = target_position_m
        self.dt = dt
        self.max_steps = max_steps
        self.seed = seed
        self.time = 0.0
        self._state = CartPoleState(
            t=0.0,
            cart_position_m=0.0,
            cart_velocity_mps=0.0,
            pole_angle_rad=0.0,
            pole_angular_velocity_radps=0.0,
        )
        self._force_command_n: float | None = None

    def get_state(self) -> CartPoleState:
        return self._state

    def minimum_phase_output(self) -> MinimumPhaseState:
        theta = self._state.pole_angle_rad
        theta_dot = self._state.pole_angular_velocity_radps
        output = self._state.cart_position_m + self.minimum_phase_gain_m * math.sin(theta)
        output_dot = self._state.cart_velocity_mps + self.minimum_phase_gain_m * theta_dot * math.cos(theta)
        return MinimumPhaseState(output_m=output, output_velocity_mps=output_dot)

    def set_force(self, newtons: float) -> None:
        try:
            force = float(newtons)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cart-pole force command must be numeric.") from exc

        if not math.isfinite(force):
            raise ValueError("Cart-pole force command must be finite.")

        self._force_command_n = max(-self.max_force_n, min(self.max_force_n, force))

    def _update(self, *, time: float, state: CartPoleState) -> None:
        self.time = time
        self._state = state

    def _clear_command(self) -> None:
        self._force_command_n = None

    def _consume_force_command(self) -> float:
        if self._force_command_n is None:
            raise ValueError("RobotPolicy.step(cart_pole) must call cart_pole.set_force(newtons).")
        return self._force_command_n


class ProfiledCart1D(ControlSystem):
    """Runtime API injected into trapezoidal motion-profile policy code."""

    def __init__(
        self,
        *,
        max_velocity_mps: float,
        max_acceleration_mps2: float,
        target_position_m: float,
        target_velocity_mps: float,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.max_velocity_mps = float(max_velocity_mps)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.target_position_m = float(target_position_m)
        self.target_velocity_mps = float(target_velocity_mps)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.time = 0.0
        self._state = MotionState1D(
            t=0.0,
            position_m=0.0,
            velocity_mps=0.0,
            acceleration_mps2=0.0,
        )
        self._acceleration_command_mps2: float | None = None

    def get_state(self) -> MotionState1D:
        return self._state

    def get_target(self) -> MotionTarget1D:
        return MotionTarget1D(
            position_m=self.target_position_m,
            velocity_mps=self.target_velocity_mps,
        )

    def limits(self) -> tuple[float, float]:
        return self.max_velocity_mps, self.max_acceleration_mps2

    def set_acceleration(self, acceleration_mps2: float) -> None:
        try:
            acceleration = float(acceleration_mps2)
        except (TypeError, ValueError) as exc:
            raise ValueError("Acceleration command must be numeric.") from exc

        if not math.isfinite(acceleration):
            raise ValueError("Acceleration command must be finite.")

        self._acceleration_command_mps2 = acceleration

    def _update(self, *, time: float, state: MotionState1D) -> None:
        self.time = float(time)
        self._state = state

    def _clear_command(self) -> None:
        self._acceleration_command_mps2 = None

    def _consume_acceleration_command(self) -> float:
        if self._acceleration_command_mps2 is None:
            raise ValueError("RobotPolicy.step(cart) must call cart.set_acceleration(acceleration_mps2).")
        return self._acceleration_command_mps2


class AccelerationEstimator1D(ControlSystem):
    """Runtime API injected into wall-distance acceleration-estimation policy code."""

    def __init__(
        self,
        *,
        wall_position_m: float,
        track_half_width_m: float,
        distance_noise_std_m: float,
        distance_quantization_m: float,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.wall_position_m = float(wall_position_m)
        self.track_half_width_m = float(track_half_width_m)
        self.distance_noise_std_m = float(distance_noise_std_m)
        self.distance_quantization_m = float(distance_quantization_m)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.time = 0.0
        self._distance_m = self.wall_position_m
        self._acceleration_estimate_mps2: float | None = None

    def distance_to_wall(self) -> float:
        return self._distance_m

    def submit_acceleration(self, acceleration_mps2: float) -> None:
        try:
            acceleration = float(acceleration_mps2)
        except (TypeError, ValueError) as exc:
            raise ValueError("Acceleration estimate must be numeric.") from exc

        if not math.isfinite(acceleration):
            raise ValueError("Acceleration estimate must be finite.")

        self._acceleration_estimate_mps2 = acceleration

    def _update(self, *, time: float, distance_m: float) -> None:
        self.time = float(time)
        self._distance_m = float(distance_m)

    def _clear_submission(self) -> None:
        self._acceleration_estimate_mps2 = None

    def _consume_acceleration_estimate(self) -> float:
        if self._acceleration_estimate_mps2 is None:
            raise ValueError("RobotPolicy.step(cart) must call cart.submit_acceleration(acceleration_mps2).")
        return self._acceleration_estimate_mps2


class DifferentialDrive(MobileRobot):
    """Runtime API injected into differential-drive policy code."""

    def __init__(
        self,
        *,
        wheel_base_m: float,
        max_wheel_velocity_mps: float,
        dt: float,
        max_steps: int,
        seed: int,
        wheel_radius_m: float = 0.06604,
        ticks_per_rev: int = 392,
    ) -> None:
        self.wheel_base_m = wheel_base_m
        self.max_wheel_velocity_mps = max_wheel_velocity_mps
        self.wheel_radius_m = wheel_radius_m
        self.ticks_per_rev = ticks_per_rev
        self.dt = dt
        self.max_steps = max_steps
        self.seed = seed
        self.time = 0.0
        self.collision_count = 0
        self._pose = Pose2d(0.0, 0.0, 0.0)
        self._target = Target2d(0.0, 0.0, 0.0)
        self._route: tuple[Pose2d, ...] = ()
        self._obstacles: tuple[CircleObstacle, ...] = ()
        self._lidar_ranges = np.array([], dtype=float)
        self._encoder_values = (0.0, 0.0)
        self._gyro_z = 0.0
        self._charger_pose: Pose2d | None = None
        self._camera_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        self._wheel_command: tuple[float, float] | None = None

    def get_pose(self) -> Pose2d:
        return self._pose

    def get_target(self) -> Target2d:
        return self._target

    def route(self) -> tuple[Pose2d, ...]:
        return self._route

    def obstacles(self) -> tuple[CircleObstacle, ...]:
        return self._obstacles

    def lidar(self) -> np.ndarray:
        return self._lidar_ranges.copy()

    def camera(self) -> np.ndarray:
        return self._camera_frame.copy()

    def get_encoder_values(self) -> tuple[float, float]:
        return self._encoder_values

    def gyro(self) -> float:
        return self._gyro_z

    def charger_pose(self) -> Pose2d | None:
        return self._charger_pose

    def set_wheel_velocity(self, left_mps: float, right_mps: float) -> None:
        try:
            left = float(left_mps)
            right = float(right_mps)
        except (TypeError, ValueError) as exc:
            raise ValueError("Wheel velocities must be numeric.") from exc

        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Wheel velocities must be finite.")

        max_velocity = self.max_wheel_velocity_mps
        self._wheel_command = (
            max(-max_velocity, min(max_velocity, left)),
            max(-max_velocity, min(max_velocity, right)),
        )

    def _update(
        self,
        *,
        time: float,
        pose: Pose2d,
        target: Target2d,
        route: tuple[Pose2d, ...],
        obstacles: tuple[CircleObstacle, ...],
        lidar_ranges: np.ndarray,
        collision_count: int,
        encoder_values: tuple[float, float] = (0.0, 0.0),
        gyro_z: float = 0.0,
        charger_pose: Pose2d | None = None,
        camera_frame: np.ndarray | None = None,
    ) -> None:
        self.time = time
        self.collision_count = collision_count
        self._pose = pose
        self._target = target
        self._route = route
        self._obstacles = obstacles
        self._lidar_ranges = lidar_ranges.copy()
        self._encoder_values = (float(encoder_values[0]), float(encoder_values[1]))
        self._gyro_z = float(gyro_z)
        self._charger_pose = charger_pose
        if camera_frame is not None:
            self._camera_frame = camera_frame.copy()

    def _clear_command(self) -> None:
        self._wheel_command = None

    def _consume_wheel_command(self) -> tuple[float, float]:
        if self._wheel_command is None:
            raise ValueError("RobotPolicy.step(robot) must command robot.set_wheel_velocity(left, right).")
        return self._wheel_command


class DifferentialDriveOdometry(MobileRobot):
    """Lower-level differential-drive API for odometry estimation tasks."""

    def __init__(
        self,
        *,
        wheel_base_m: float,
        wheel_radius_m: float,
        max_wheel_velocity_mps: float,
        ticks_per_rev: int,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.wheel_base_m = wheel_base_m
        self.wheel_radius_m = wheel_radius_m
        self.max_wheel_velocity_mps = max_wheel_velocity_mps
        self.ticks_per_rev = ticks_per_rev
        self.dt = dt
        self.max_steps = max_steps
        self.seed = seed
        self.time = 0.0
        self.collision_count = 0
        self._encoder_values = (0, 0)
        self._gyro_z = 0.0
        self._odometry_estimate: Pose2d | None = None

    def get_encoder_values(self) -> tuple[int, int]:
        return self._encoder_values

    def gyro(self) -> float:
        return self._gyro_z

    def submit_odometry(self, pose: Pose2d) -> None:
        try:
            x = float(pose.x)
            y = float(pose.y)
            yaw = float(pose.yaw)
        except AttributeError as exc:
            raise ValueError("submit_odometry expects a Pose2d(x, y, yaw) estimate.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("Odometry estimate fields must be numeric.") from exc

        if not all(math.isfinite(value) for value in (x, y, yaw)):
            raise ValueError("Odometry estimate fields must be finite.")
        self._odometry_estimate = Pose2d(x=x, y=y, yaw=math.atan2(math.sin(yaw), math.cos(yaw)))

    def _update(
        self,
        *,
        time: float,
        encoder_values: tuple[int, int],
        gyro_z: float,
        collision_count: int,
    ) -> None:
        self.time = time
        self.collision_count = collision_count
        self._encoder_values = (int(encoder_values[0]), int(encoder_values[1]))
        self._gyro_z = float(gyro_z)

    def _clear_step_outputs(self) -> None:
        self._odometry_estimate = None

    def _consume_odometry_estimate(self) -> Pose2d:
        if self._odometry_estimate is None:
            raise ValueError("RobotPolicy.step(robot) must call robot.submit_odometry(Pose2d(x, y, yaw)).")
        return self._odometry_estimate


class DifferentialDriveStateEstimator(DifferentialDriveOdometry):
    """Differential-drive API for fusing wheel odometry with AprilTag pose updates."""

    def __init__(
        self,
        *,
        wheel_base_m: float,
        wheel_radius_m: float,
        max_wheel_velocity_mps: float,
        ticks_per_rev: int,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        super().__init__(
            wheel_base_m=wheel_base_m,
            wheel_radius_m=wheel_radius_m,
            max_wheel_velocity_mps=max_wheel_velocity_mps,
            ticks_per_rev=ticks_per_rev,
            dt=dt,
            max_steps=max_steps,
            seed=seed,
        )
        self._april_tag_measurements: tuple[AprilTagPoseEstimate, ...] = ()

    def april_tag_measurements(self) -> tuple[AprilTagPoseEstimate, ...]:
        return self._april_tag_measurements

    def _update(
        self,
        *,
        time: float,
        encoder_values: tuple[int, int],
        gyro_z: float,
        collision_count: int,
        april_tag_measurements: tuple[AprilTagPoseEstimate, ...] = (),
    ) -> None:
        super()._update(
            time=time,
            encoder_values=encoder_values,
            gyro_z=gyro_z,
            collision_count=collision_count,
        )
        self._april_tag_measurements = tuple(april_tag_measurements)


class DifferentialDriveSlam(DifferentialDriveOdometry):
    """Differential-drive API for 2D lidar SLAM map-submission tasks."""

    def __init__(
        self,
        *,
        wheel_base_m: float,
        wheel_radius_m: float,
        max_wheel_velocity_mps: float,
        ticks_per_rev: int,
        dt: float,
        max_steps: int,
        seed: int,
        lidar_angle_min_rad: float,
        lidar_angle_increment_rad: float,
        lidar_max_range_m: float,
        max_map_points: int,
    ) -> None:
        super().__init__(
            wheel_base_m=wheel_base_m,
            wheel_radius_m=wheel_radius_m,
            max_wheel_velocity_mps=max_wheel_velocity_mps,
            ticks_per_rev=ticks_per_rev,
            dt=dt,
            max_steps=max_steps,
            seed=seed,
        )
        self.lidar_angle_min_rad = float(lidar_angle_min_rad)
        self.lidar_angle_increment_rad = float(lidar_angle_increment_rad)
        self.lidar_max_range_m = float(lidar_max_range_m)
        self.max_map_points = int(max_map_points)
        self._lidar_ranges = np.array([], dtype=float)
        self._slam_pose: Pose2d | None = None
        self._slam_map_points: tuple[MapPoint2d, ...] = ()

    def lidar(self) -> np.ndarray:
        return self._lidar_ranges.copy()

    def lidar_angles(self) -> np.ndarray:
        if len(self._lidar_ranges) == 0:
            return np.array([], dtype=float)
        return self.lidar_angle_min_rad + np.arange(len(self._lidar_ranges), dtype=float) * self.lidar_angle_increment_rad

    def submit_slam(
        self,
        pose: Pose2d,
        map_points: list[MapPoint2d] | tuple[MapPoint2d, ...] | list[tuple[float, float]],
    ) -> None:
        try:
            x = float(pose.x)
            y = float(pose.y)
            yaw = float(pose.yaw)
        except AttributeError as exc:
            raise ValueError("submit_slam expects a Pose2d(x, y, yaw) estimate.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("SLAM pose estimate fields must be numeric.") from exc

        if not all(math.isfinite(value) for value in (x, y, yaw)):
            raise ValueError("SLAM pose estimate fields must be finite.")

        parsed_points: list[MapPoint2d] = []
        for point in map_points:
            try:
                point_x = float(point.x)  # type: ignore[attr-defined]
                point_y = float(point.y)  # type: ignore[attr-defined]
            except AttributeError:
                try:
                    if isinstance(point, dict):
                        point_x = float(point["x"])
                        point_y = float(point["y"])
                    else:
                        point_x = float(point[0])  # type: ignore[index]
                        point_y = float(point[1])  # type: ignore[index]
                except (TypeError, ValueError, IndexError, KeyError) as exc:
                    raise ValueError("map_points must contain MapPoint2d objects or (x, y) pairs.") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError("Map point fields must be numeric.") from exc

            if math.isfinite(point_x) and math.isfinite(point_y):
                parsed_points.append(MapPoint2d(x=point_x, y=point_y))
            if len(parsed_points) >= self.max_map_points:
                break

        self._slam_pose = Pose2d(x=x, y=y, yaw=math.atan2(math.sin(yaw), math.cos(yaw)))
        self._slam_map_points = tuple(parsed_points)

    def _update(
        self,
        *,
        time: float,
        encoder_values: tuple[int, int],
        gyro_z: float,
        collision_count: int,
        lidar_ranges: np.ndarray,
    ) -> None:
        super()._update(
            time=time,
            encoder_values=encoder_values,
            gyro_z=gyro_z,
            collision_count=collision_count,
        )
        self._lidar_ranges = np.asarray(lidar_ranges, dtype=float).copy()

    def _clear_step_outputs(self) -> None:
        self._slam_pose = None
        self._slam_map_points = ()

    def _consume_slam_submission(self) -> tuple[Pose2d, tuple[MapPoint2d, ...]]:
        if self._slam_pose is None:
            raise ValueError("RobotPolicy.step(robot) must call robot.submit_slam(Pose2d(...), map_points).")
        return self._slam_pose, self._slam_map_points


class Quadrotor(AerialRobot):
    """Runtime API injected into quadrotor gate-navigation policy code."""

    def __init__(
        self,
        *,
        hover_power: float,
        max_power: float,
        max_body_rate_radps: float,
        max_tilt_rad: float,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.hover_power = hover_power
        self.max_power = max_power
        self.max_body_rate_radps = max_body_rate_radps
        self.max_tilt_rad = max_tilt_rad
        self.dt = dt
        self.max_steps = max_steps
        self.seed = seed
        self.time = 0.0
        self.collision_count = 0
        self._pose = Pose3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._next_gate = Gate3d("gate_0", 0.0, 0.0, 1.0, 0.0, 1.0, 0.7)
        self._gates_completed = 0
        self._gate_count = 0
        self._command: tuple[float, float, float, float] | None = None

    def get_pose(self) -> Pose3d:
        return self._pose

    def get_altitude(self) -> float:
        return self._pose.z

    def get_next_gate(self) -> Gate3d:
        return self._next_gate

    def gates_completed(self) -> int:
        return self._gates_completed

    def gate_count(self) -> int:
        return self._gate_count

    def set_body_rate_and_power(
        self,
        roll_rate_radps: float,
        pitch_rate_radps: float,
        yaw_rate_radps: float,
        power: float,
    ) -> None:
        try:
            roll_rate = float(roll_rate_radps)
            pitch_rate = float(pitch_rate_radps)
            yaw_rate = float(yaw_rate_radps)
            collective_power = float(power)
        except (TypeError, ValueError) as exc:
            raise ValueError("Body rates and power must be numeric.") from exc

        values = (roll_rate, pitch_rate, yaw_rate, collective_power)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Body rates and power must be finite.")

        rate_limit = self.max_body_rate_radps
        self._command = (
            max(-rate_limit, min(rate_limit, roll_rate)),
            max(-rate_limit, min(rate_limit, pitch_rate)),
            max(-rate_limit, min(rate_limit, yaw_rate)),
            max(0.0, min(self.max_power, collective_power)),
        )

    def _update(
        self,
        *,
        time: float,
        pose: Pose3d,
        next_gate: Gate3d,
        gates_completed: int,
        gate_count: int,
        collision_count: int,
    ) -> None:
        self.time = time
        self.collision_count = collision_count
        self._pose = pose
        self._next_gate = next_gate
        self._gates_completed = gates_completed
        self._gate_count = gate_count

    def _clear_command(self) -> None:
        self._command = None

    def _consume_command(self) -> tuple[float, float, float, float]:
        if self._command is None:
            raise ValueError(
                "RobotPolicy.step(robot) must command "
                "robot.set_body_rate_and_power(roll_rate, pitch_rate, yaw_rate, power)."
            )
        return self._command


class CollisionProbe(MobileRobot):
    """Runtime API injected into IMU collision-detection policy code."""

    severity_levels = ("none", "light", "hard")

    def __init__(
        self,
        *,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.dt = dt
        self.max_steps = max_steps
        self.seed = seed
        self.time = 0.0
        self._imu_sample = ImuSample(t=0.0, ax=0.0, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.0)
        self._decision: CollisionDecision | None = None

    def imu(self) -> ImuSample:
        return self._imu_sample

    def submit_collision_decision(self, contact: bool, severity: str) -> None:
        if not isinstance(contact, bool):
            raise ValueError("contact must be a bool.")
        if severity not in self.severity_levels:
            allowed = ", ".join(self.severity_levels)
            raise ValueError(f"severity must be one of: {allowed}.")
        self._decision = CollisionDecision(contact=contact, severity=severity)

    def _update(self, *, time: float, imu_sample: ImuSample) -> None:
        self.time = time
        self._imu_sample = imu_sample

    def _clear_decision(self) -> None:
        self._decision = None

    def _consume_collision_decision(self) -> CollisionDecision:
        if self._decision is None:
            raise ValueError("RobotPolicy.step(robot) must call robot.submit_collision_decision(contact, severity).")
        return self._decision


class CurrentControlledMotor(Actuator):
    """Runtime API injected into motor torque-control policy code."""

    def __init__(
        self,
        *,
        shaft_length_m: float,
        kt_nm_per_amp: float,
        max_current_a: float,
        target_force_n: float,
        dt: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.shaft_length_m = shaft_length_m
        self.kt_nm_per_amp = kt_nm_per_amp
        self.max_current_a = max_current_a
        self.target_force_n = target_force_n
        self.dt = dt
        self.max_steps = max_steps
        self.seed = seed
        self.time = 0.0
        self._measured_current_a = 0.0
        self._scale_force_n = 0.0
        self._current_command_a: float | None = None

    def target_force(self) -> float:
        return self.target_force_n

    def current(self) -> float:
        return self._measured_current_a

    def scale_force(self) -> float:
        return self._scale_force_n

    def set_current(self, amps: float) -> None:
        try:
            current = float(amps)
        except (TypeError, ValueError) as exc:
            raise ValueError("Motor current command must be numeric.") from exc

        if not math.isfinite(current):
            raise ValueError("Motor current command must be finite.")

        self._current_command_a = max(-self.max_current_a, min(self.max_current_a, current))

    def _update(self, *, time: float, measured_current_a: float, scale_force_n: float) -> None:
        self.time = time
        self._measured_current_a = measured_current_a
        self._scale_force_n = scale_force_n

    def _clear_command(self) -> None:
        self._current_command_a = None

    def _consume_current_command(self) -> float:
        if self._current_command_a is None:
            raise ValueError("RobotPolicy.step(motor) must call motor.set_current(amps).")
        return self._current_command_a


@runtime_checkable
class RobotPolicyProtocol(Protocol):
    """Protocol implemented by submitted policy classes."""

    def step(self, robot: object) -> None:
        ...


class RobotPolicy:
    """Reference API for challenge submissions.

    User modules should define their own class with this name and these methods.
    The platform instantiates it with no arguments.
    """

    def step(self, robot: object) -> None:
        raise NotImplementedError
