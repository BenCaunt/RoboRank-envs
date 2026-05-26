from __future__ import annotations

import math

from roborank_envs.models import (
    CameraSpec,
    CartPoleSpec,
    DifferentialDriveRobotSpec,
    LidarSpec,
    ManipulatorSpec,
    MotorStandSpec,
    ProfiledCartSpec,
    QuadrotorSpec,
)


INCH_TO_M = 0.0254


def differential_drive_cube_v1() -> DifferentialDriveRobotSpec:
    """Reusable 12 inch-class differential-drive mobile robot primitive."""

    footprint = 12.0 * INCH_TO_M
    height = 12.0 * INCH_TO_M
    wheel_base = 10.5 * INCH_TO_M
    wheel_radius = 2.6 * INCH_TO_M
    wheel_width = 1.4 * INCH_TO_M

    return DifferentialDriveRobotSpec(
        type="differential_drive",
        model="differential_drive_cube_v1",
        footprint_length_m=footprint,
        footprint_width_m=footprint,
        height_m=height,
        wheel_base_m=wheel_base,
        wheel_radius_m=wheel_radius,
        wheel_width_m=wheel_width,
        radius_m=math.hypot(footprint / 2, footprint / 2),
        max_wheel_velocity_mps=1.25,
        cameras=[
            CameraSpec(
                name="front_camera",
                frame="front_camera",
                fov_y_deg=120.0,
                width=384,
                height=240,
                xyz_m={
                    "x": footprint / 2 + 0.012,
                    "y": 0.0,
                    "z": height * 0.58,
                },
                xyaxes="0 -1 0 0 0 1",
            )
        ],
        lidars=[
            LidarSpec(
                name="lidar_2d_top",
                frame="lidar_2d_top",
                fov_deg=360.0,
                num_rays=360,
                max_range_m=6.0,
                xyz_m={
                    "x": 0.0,
                    "y": 0.0,
                    "z": height + 0.018,
                },
            )
        ],
    )


def motor_stand_scale_v1() -> MotorStandSpec:
    """Bench fixture with a current-controlled motor pressing a small scale."""

    return MotorStandSpec(
        type="motor_stand",
        model="motor_stand_scale_v1",
        shaft_length_m=0.18,
        shaft_radius_m=0.008,
        kt_nm_per_amp=0.041,
        max_current_a=6.0,
        current_time_constant_sec=0.08,
        scale_time_constant_sec=0.12,
        scale_stiffness_n_per_m=1400.0,
        max_force_n=1.35,
        base_length_m=0.36,
        base_width_m=0.20,
        base_height_m=0.035,
        scale_plate_width_m=0.13,
        scale_plate_depth_m=0.11,
    )


def quadrotor_x500_v1() -> QuadrotorSpec:
    """Small quadrotor primitive for gate-course control challenges."""

    arm_length = 0.225
    body_length = 0.18
    body_width = 0.12
    height = 0.07

    return QuadrotorSpec(
        type="quadrotor",
        model="quadrotor_x500_v1",
        footprint_length_m=arm_length * 2.0,
        footprint_width_m=arm_length * 2.0,
        height_m=height,
        radius_m=0.16,
        mass_kg=1.15,
        arm_length_m=arm_length,
        hover_power=0.5,
        max_power=1.0,
        max_body_rate_radps=2.8,
        max_tilt_rad=math.radians(55.0),
        cameras=[
            CameraSpec(
                name="front_camera",
                frame="front_camera",
                fov_y_deg=95.0,
                width=384,
                height=240,
                xyz_m={
                    "x": body_length / 2 + 0.025,
                    "y": 0.0,
                    "z": 0.0,
                },
                xyaxes="0 -1 0 0 0 1",
            )
        ],
    )


def cart_pole_v1() -> CartPoleSpec:
    """Classic cart-pole plant with explicit physical constants."""

    return CartPoleSpec(
        type="cart_pole",
        model="cart_pole_v1",
        cart_mass_kg=1.0,
        pole_mass_kg=0.1,
        pole_com_length_m=0.5,
        pole_length_m=1.0,
        pole_radius_m=0.012,
        cart_width_m=0.30,
        cart_height_m=0.18,
        track_half_width_m=2.4,
        max_force_n=10.0,
        gravity_mps2=9.81,
        minimum_phase_gain_m=0.75,
    )


def profiled_cart_1d_v1() -> ProfiledCartSpec:
    """One-dimensional cart fixture for time-optimal motion-profile challenges."""

    return ProfiledCartSpec(
        type="profiled_cart_1d",
        model="profiled_cart_1d_v1",
        cart_mass_kg=1.2,
        cart_width_m=0.28,
        cart_height_m=0.16,
        track_half_width_m=4.2,
        max_velocity_mps=1.25,
        max_acceleration_mps2=1.6,
    )


def planar_2link_arm_v1() -> ManipulatorSpec:
    """Two revolute joints moving in the horizontal XY plane."""

    link_1 = 0.42
    link_2 = 0.34
    return ManipulatorSpec(
        type="manipulator",
        model="planar_2link_arm_v1",
        mechanism="planar_2link",
        joint_count=2,
        link_lengths_m=[link_1, link_2],
        joint_limits_rad=[
            [-math.pi, math.pi],
            [0.0, math.radians(165.0)],
        ],
        end_effector_radius_m=0.022,
        workspace_radius_m=link_1 + link_2,
    )


def turret_2link_arm_v1() -> ManipulatorSpec:
    """Yaw turret plus two pitch joints reaching points in 3D space."""

    link_1 = 0.46
    link_2 = 0.38
    return ManipulatorSpec(
        type="manipulator",
        model="turret_2link_arm_v1",
        mechanism="turret_2link",
        joint_count=3,
        link_lengths_m=[link_1, link_2],
        joint_limits_rad=[
            [-math.pi, math.pi],
            [math.radians(-35.0), math.radians(80.0)],
            [math.radians(-145.0), math.radians(125.0)],
        ],
        base_height_m=0.18,
        end_effector_radius_m=0.024,
        workspace_radius_m=link_1 + link_2,
    )


def five_bar_scara_v1() -> ManipulatorSpec:
    """Planar two-motor, closed-chain drawing mechanism."""

    proximal = 0.30
    distal = 0.34
    base_spacing = 0.36
    return ManipulatorSpec(
        type="manipulator",
        model="five_bar_scara_v1",
        mechanism="five_bar_scara",
        joint_count=2,
        link_lengths_m=[proximal, distal],
        joint_limits_rad=[
            [math.radians(5.0), math.radians(160.0)],
            [math.radians(20.0), math.radians(178.0)],
        ],
        base_spacing_m=base_spacing,
        end_effector_radius_m=0.018,
        workspace_radius_m=0.62,
    )
