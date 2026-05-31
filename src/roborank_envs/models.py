from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class Pose2D(BaseModel):
    x: float
    y: float
    yaw: float


class PoseSample(Pose2D):
    t: float
    z: float | None = None
    roll: float | None = None
    pitch: float | None = None
    vx: float | None = None
    vy: float | None = None
    vz: float | None = None
    speed: float | None = None


class OdometryEstimateSample(Pose2D):
    t: float
    frame_index: int
    position_error_m: float
    yaw_error_rad: float


class MapPoint2D(BaseModel):
    x: float
    y: float


class AprilTagPoseSample(Pose2D):
    t: float
    capture_t: float
    frame_index: int
    tag_id: int
    distance_m: float
    bearing_rad: float
    position_std_m: float
    yaw_std_rad: float
    latency_sec: float
    ambiguity: float


class Target(BaseModel):
    x: float
    y: float
    radius: float


class Obstacle(BaseModel):
    id: str
    x: float
    y: float
    radius: float


class CameraSpec(BaseModel):
    name: str
    frame: str
    fov_y_deg: float
    width: int
    height: int
    xyz_m: dict[str, float]
    xyaxes: str


class LidarSpec(BaseModel):
    name: str
    frame: str
    fov_deg: float
    num_rays: int
    max_range_m: float
    xyz_m: dict[str, float]


class DifferentialDriveRobotSpec(BaseModel):
    type: Literal["differential_drive"]
    model: str = "differential_drive_cube_v1"
    footprint_length_m: float
    footprint_width_m: float
    height_m: float
    wheel_base_m: float
    wheel_radius_m: float
    wheel_width_m: float
    radius_m: float
    max_wheel_velocity_mps: float
    cameras: list[CameraSpec] = Field(default_factory=list)
    lidars: list[LidarSpec] = Field(default_factory=list)


class MotorStandSpec(BaseModel):
    type: Literal["motor_stand"]
    model: str = "motor_stand_scale_v1"
    shaft_length_m: float
    shaft_radius_m: float
    kt_nm_per_amp: float
    max_current_a: float
    current_time_constant_sec: float
    scale_time_constant_sec: float
    scale_stiffness_n_per_m: float
    max_force_n: float
    base_length_m: float
    base_width_m: float
    base_height_m: float
    scale_plate_width_m: float
    scale_plate_depth_m: float


class QuadrotorSpec(BaseModel):
    type: Literal["quadrotor"]
    model: str = "quadrotor_x500_v1"
    footprint_length_m: float
    footprint_width_m: float
    height_m: float
    radius_m: float
    mass_kg: float
    arm_length_m: float
    hover_power: float
    max_power: float
    max_body_rate_radps: float
    max_tilt_rad: float
    cameras: list[CameraSpec] = Field(default_factory=list)


class CartPoleSpec(BaseModel):
    type: Literal["cart_pole"]
    model: str = "cart_pole_v1"
    cart_mass_kg: float
    pole_mass_kg: float
    pole_com_length_m: float
    pole_length_m: float
    pole_radius_m: float
    cart_width_m: float
    cart_height_m: float
    track_half_width_m: float
    max_force_n: float
    gravity_mps2: float
    minimum_phase_gain_m: float


class ProfiledCartSpec(BaseModel):
    type: Literal["profiled_cart_1d"]
    model: str = "profiled_cart_1d_v1"
    cart_mass_kg: float
    cart_width_m: float
    cart_height_m: float
    track_half_width_m: float
    max_velocity_mps: float
    max_acceleration_mps2: float


class ManipulatorSpec(BaseModel):
    type: Literal["manipulator"]
    model: str
    mechanism: Literal["planar_2link", "turret_2link", "five_bar_scara"]
    joint_count: int
    link_lengths_m: list[float]
    joint_limits_rad: list[list[float]]
    base_height_m: float = 0.0
    base_spacing_m: float | None = None
    end_effector_radius_m: float = 0.025
    workspace_radius_m: float = 1.0


RobotSpec = DifferentialDriveRobotSpec | MotorStandSpec | QuadrotorSpec | CartPoleSpec | ProfiledCartSpec | ManipulatorSpec


class ChallengeSummary(BaseModel):
    id: str
    title: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    robot: str
    description: str


class ChallengeSpec(BaseModel):
    id: str
    title: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    description: str
    robot: RobotSpec
    sensors: list[str]
    actuators: list[str]
    objective: str
    success_conditions: dict[str, Any]
    scoring: dict[str, Any]
    defaults: dict[str, Any]

    def summary(self) -> ChallengeSummary:
        return ChallengeSummary(
            id=self.id,
            title=self.title,
            difficulty=self.difficulty,
            robot=self.robot.type,
            description=self.description,
        )


class CollisionEvent(BaseModel):
    t: float
    kind: Literal["obstacle", "bounds"]
    object_id: str
    penetration_m: float


class ControlSample(BaseModel):
    t: float
    left_wheel_velocity: float | None = None
    right_wheel_velocity: float | None = None
    roll_rate_radps: float | None = None
    pitch_rate_radps: float | None = None
    yaw_rate_radps: float | None = None
    power: float | None = None


class MotorControlSample(BaseModel):
    t: float
    current_command_a: float


class MotorStateSample(BaseModel):
    t: float
    frame_index: int
    current_command_a: float
    measured_current_a: float
    motor_torque_nm: float
    shaft_angle_rad: float
    shaft_tip_deflection_m: float
    scale_force_n: float
    target_force_n: float
    force_error_n: float


class CartPoleControlSample(BaseModel):
    t: float
    force_n: float


class CartPoleStateSample(BaseModel):
    t: float
    frame_index: int
    cart_position_m: float
    cart_velocity_mps: float
    pole_angle_rad: float
    pole_angular_velocity_radps: float
    target_position_m: float
    minimum_phase_output_m: float | None = None
    minimum_phase_output_velocity_mps: float | None = None
    force_n: float | None = None


class MotionProfileControlSample(BaseModel):
    t: float
    acceleration_command_mps2: float
    applied_acceleration_mps2: float
    acceleration_limit_violation: bool = False


class MotionProfileStateSample(BaseModel):
    t: float
    frame_index: int
    position_m: float
    velocity_mps: float
    acceleration_mps2: float
    target_position_m: float
    target_velocity_mps: float
    position_error_m: float
    velocity_error_mps: float
    acceleration_command_mps2: float | None = None
    acceleration_limit_violation: bool = False
    velocity_limit_violation: bool = False


class AccelerationEstimateSample(BaseModel):
    t: float
    frame_index: int
    position_m: float
    velocity_mps: float
    acceleration_mps2: float
    wall_position_m: float
    measured_distance_m: float
    measured_position_m: float
    estimated_acceleration_mps2: float
    acceleration_error_mps2: float


class InverseKinematicsSample(BaseModel):
    t: float
    frame_index: int
    target_x: float
    target_y: float
    target_z: float
    actual_x: float
    actual_y: float
    actual_z: float
    position_error_m: float
    joint_angles_rad: list[float]
    joint_limit_violation: bool = False


class RenderFrame(BaseModel):
    t: float
    frame_index: int
    image_data_url: str
    width: int
    height: int
    camera: str


class LidarScanSample(BaseModel):
    t: float
    frame_index: int
    frame: str
    angles_rad: list[float]
    ranges_m: list[float]
    max_range_m: float


class EncoderTraceSample(BaseModel):
    t: float
    frame_index: int
    left_ticks: int
    right_ticks: int


class GyroTraceSample(BaseModel):
    t: float
    frame_index: int
    yaw_rate_radps: float


class ChargerPoseSample(BaseModel):
    t: float
    frame_index: int
    visible: bool
    x: float | None = None
    y: float | None = None
    yaw: float | None = None


class ImuTraceSample(BaseModel):
    t: float
    frame_index: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class CollisionDecisionSample(BaseModel):
    t: float
    frame_index: int
    contact: bool
    severity: Literal["none", "light", "hard"]


class ReplayArtifact(BaseModel):
    type: Literal["rerun_rrd"]
    name: str
    url: str
    mime_type: str = "application/octet-stream"


class ScoreMetrics(BaseModel):
    metric_kind: Literal[
        "navigation",
        "odometry",
        "force_control",
        "gate_sequence",
        "cart_pole",
        "motion_profile",
        "acceleration_estimation",
        "inverse_kinematics",
        "slam",
    ] = "navigation"
    score: float
    success: bool
    status: Literal[
        "success",
        "collision",
        "timeout",
        "error",
        "overshoot",
        "force_error",
        "accuracy_error",
        "limit_violation",
    ]
    elapsed_sec: float
    distance_to_target_m: float
    heading_error_rad: float | None = None
    collision_count: int
    path_length_m: float
    energy_used: float
    smoothness_cost: float
    target_force_n: float | None = None
    final_force_error_n: float | None = None
    mean_abs_force_error_n: float | None = None
    settling_time_sec: float | None = None
    overshoot_pct: float | None = None
    peak_force_n: float | None = None
    gates_completed: int | None = None
    gate_count: int | None = None
    max_attitude_deg: float | None = None
    final_position_error_m: float | None = None
    mean_position_error_m: float | None = None
    max_position_error_m: float | None = None
    final_yaw_error_rad: float | None = None
    mean_yaw_error_rad: float | None = None
    excitation_distance_m: float | None = None
    excitation_yaw_rad: float | None = None
    final_pole_angle_rad: float | None = None
    max_abs_pole_angle_rad: float | None = None
    rms_pole_angle_rad: float | None = None
    final_cart_position_m: float | None = None
    max_abs_cart_position_m: float | None = None
    final_minimum_phase_output_m: float | None = None
    rms_minimum_phase_output_m: float | None = None
    final_velocity_error_mps: float | None = None
    max_abs_velocity_mps: float | None = None
    max_abs_acceleration_command_mps2: float | None = None
    acceleration_limit_violation_count: int | None = None
    velocity_limit_violation_count: int | None = None
    optimal_time_sec: float | None = None
    finish_time_sec: float | None = None
    acceleration_rmse_mps2: float | None = None
    mean_abs_acceleration_error_mps2: float | None = None
    final_acceleration_error_mps2: float | None = None
    phase_lag_sec: float | None = None
    acceleration_correlation: float | None = None
    derivative_baseline_rmse_mps2: float | None = None
    moving_average_baseline_rmse_mps2: float | None = None
    target_count: int | None = None
    joint_limit_violation_count: int | None = None
    map_chamfer_error_m: float | None = None
    map_coverage_ratio: float | None = None
    map_false_positive_ratio: float | None = None
    map_point_count: int | None = None


class ReplayTrace(BaseModel):
    poses: list[PoseSample] = Field(default_factory=list)
    target: Target = Field(default_factory=lambda: Target(x=0.0, y=0.0, radius=0.0))
    obstacles: list[Obstacle] = Field(default_factory=list)
    collisions: list[CollisionEvent] = Field(default_factory=list)
    controls: list[ControlSample] = Field(default_factory=list)
    motor_controls: list[MotorControlSample] = Field(default_factory=list)
    motor_states: list[MotorStateSample] = Field(default_factory=list)
    cart_pole_controls: list[CartPoleControlSample] = Field(default_factory=list)
    cart_pole_states: list[CartPoleStateSample] = Field(default_factory=list)
    motion_profile_controls: list[MotionProfileControlSample] = Field(default_factory=list)
    motion_profile_states: list[MotionProfileStateSample] = Field(default_factory=list)
    acceleration_estimates: list[AccelerationEstimateSample] = Field(default_factory=list)
    inverse_kinematics_samples: list[InverseKinematicsSample] = Field(default_factory=list)
    render_frames: list[RenderFrame] = Field(default_factory=list)
    lidar_scans: list[LidarScanSample] = Field(default_factory=list)
    encoder_samples: list[EncoderTraceSample] = Field(default_factory=list)
    gyro_samples: list[GyroTraceSample] = Field(default_factory=list)
    odometry_estimates: list[OdometryEstimateSample] = Field(default_factory=list)
    slam_map_points: list[MapPoint2D] = Field(default_factory=list)
    april_tag_pose_samples: list[AprilTagPoseSample] = Field(default_factory=list)
    charger_pose_samples: list[ChargerPoseSample] = Field(default_factory=list)
    imu_samples: list[ImuTraceSample] = Field(default_factory=list)
    collision_decisions: list[CollisionDecisionSample] = Field(default_factory=list)
    artifacts: list[ReplayArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    challenge_id: str = Field(
        default="diff_drive_reach_target",
        validation_alias=AliasChoices("challenge_id", "challengeId"),
    )
    language: Literal["python"] = "python"
    policy_source: str = Field(
        validation_alias=AliasChoices("policy_source", "code"),
        description="Python source containing a RobotPolicy class.",
        max_length=200_000,
    )
    max_steps: int | None = Field(default=None, ge=1, le=5000)

    @model_validator(mode="before")
    @classmethod
    def reject_local_policy_imports(cls, data: object) -> object:
        if isinstance(data, dict) and any(key in data for key in ("policy_path", "policy_module")):
            raise ValueError("policy_path and policy_module are not accepted by the run API. Submit policy_source/code.")
        return data

    @field_validator("policy_source")
    @classmethod
    def policy_source_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy_source/code must not be empty.")
        return value


class RunResult(BaseModel):
    challenge_id: str
    seed: int
    metrics: ScoreMetrics
    replay: ReplayTrace
    message: str | None = None
    logs: list[str] = Field(default_factory=list)
