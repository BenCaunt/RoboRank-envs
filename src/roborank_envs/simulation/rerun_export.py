from __future__ import annotations

import base64
import math
import os
import re
import uuid
from pathlib import Path
from typing import Iterable

from roborank_envs.models import ChallengeSpec, PoseSample, RenderFrame, ReplayArtifact, ReplayTrace, ScoreMetrics


RUN_ARTIFACT_DIR = Path(os.environ.get("ROBORANK_ARTIFACT_DIR", Path.cwd() / "runs"))


def write_rerun_recording(
    *,
    challenge: ChallengeSpec,
    seed: int,
    replay: ReplayTrace,
    metrics: ScoreMetrics,
    artifact_dir: Path | None = None,
    artifact_prefix: str | None = None,
) -> tuple[ReplayArtifact | None, str | None]:
    if os.environ.get("ROBORANK_DISABLE_RERUN_EXPORT") == "1":
        return None, None
    try:
        import rerun as rr
    except Exception as exc:  # noqa: BLE001 - Rerun should not be required for grading.
        return None, f"{type(exc).__name__}: {exc}"

    output_dir = artifact_dir or RUN_ARTIFACT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _artifact_filename(challenge_id=challenge.id, seed=seed, artifact_prefix=artifact_prefix)
    output_path = output_dir / filename
    if output_path.exists():
        output_path.unlink()

    try:
        rec = rr.RecordingStream("roborank_local", recording_id=challenge.id)
        rec.save(output_path)

        _log_static_scene(rr=rr, rec=rec, challenge=challenge, replay=replay, metrics=metrics)
        _log_time_series(rr=rr, rec=rec, challenge=challenge, replay=replay)

        rec.flush()
        rec.disconnect()
    except Exception as exc:  # noqa: BLE001 - Rerun export is diagnostic, not grading-critical.
        return None, f"{type(exc).__name__}: {exc}"

    return (
        ReplayArtifact(
            type="rerun_rrd",
            name="Rerun recording",
            url=f"/artifacts/{filename}",
        ),
        None,
    )


def _artifact_filename(*, challenge_id: str, seed: int, artifact_prefix: str | None) -> str:
    slug = _safe_slug(challenge_id)
    if artifact_prefix:
        return f"{_safe_slug(artifact_prefix)}_{slug}.rrd"
    if os.environ.get("ROBORANK_UNIQUE_ARTIFACTS") == "1":
        return f"{slug}_{uuid.uuid4().hex[:12]}.rrd"
    return f"{slug}.rrd"


def _log_static_scene(*, rr: object, rec: object, challenge: ChallengeSpec, replay: ReplayTrace, metrics: ScoreMetrics) -> None:
    if challenge.robot.type == "profiled_cart_1d":
        _log_motion_profile_static_scene(rr=rr, rec=rec, challenge=challenge, replay=replay, metrics=metrics)
        return
    if challenge.robot.type == "cart_pole":
        _log_cart_pole_static_scene(rr=rr, rec=rec, challenge=challenge, replay=replay, metrics=metrics)
        return
    if challenge.robot.type == "manipulator":
        _log_manipulator_static_scene(rr=rr, rec=rec, challenge=challenge, replay=replay, metrics=metrics)
        return
    if challenge.robot.type == "motor_stand":
        _log_motor_static_scene(rr=rr, rec=rec, challenge=challenge, replay=replay, metrics=metrics)
        return
    if challenge.robot.type == "quadrotor":
        _log_quadrotor_static_scene(rr=rr, rec=rec, challenge=challenge, replay=replay, metrics=metrics)
        return

    bounds = replay.metadata.get("arena_bounds_m", {})
    robot = challenge.robot

    rec.log(  # type: ignore[attr-defined]
        "world/arena_bounds",
        rr.LineStrips3D([_bounds_strip(bounds)], radii=0.012, colors=[(92, 105, 125, 255)]),  # type: ignore[attr-defined]
        static=True,
    )
    rec.log(  # type: ignore[attr-defined]
        "world/target",
        rr.LineStrips3D(  # type: ignore[attr-defined]
            [_circle_strip(replay.target.x, replay.target.y, replay.target.radius, z=0.025)],
            radii=0.015,
            colors=[(33, 104, 105, 255)],
        ),
        static=True,
    )
    target_yaw = replay.metadata.get("target_yaw_rad")
    if isinstance(target_yaw, (float, int)):
        rec.log(  # type: ignore[attr-defined]
            "world/target_heading",
            rr.Arrows3D(  # type: ignore[attr-defined]
                vectors=[[0.28 * math.cos(target_yaw), 0.28 * math.sin(target_yaw), 0.0]],
                origins=[[replay.target.x, replay.target.y, 0.08]],
                colors=[(33, 104, 105, 255)],
            ),
            static=True,
        )
    for obstacle in replay.obstacles:
        rec.log(  # type: ignore[attr-defined]
            f"world/obstacles/{obstacle.id}",
            rr.LineStrips3D(  # type: ignore[attr-defined]
                [_circle_strip(obstacle.x, obstacle.y, obstacle.radius, z=robot.height_m / 2)],
                radii=0.02,
                colors=[(176, 79, 72, 255)],
            ),
            static=True,
        )

    route = _route_strip(replay.metadata.get("route_waypoints"))
    if len(route) >= 2:
        rec.log(  # type: ignore[attr-defined]
            "world/route",
            rr.LineStrips3D([route], radii=0.018, colors=[(47, 112, 186, 255)]),  # type: ignore[attr-defined]
            static=True,
        )

    tag_layout = _tag_layout_points(replay.metadata.get("april_tag_layout"), z=robot.height_m + 0.04)
    if tag_layout:
        rec.log(  # type: ignore[attr-defined]
            "world/apriltag_layout",
            rr.Points3D(tag_layout, radii=0.045, colors=(24, 30, 38, 255)),  # type: ignore[attr-defined]
            static=True,
        )

    rec.log(  # type: ignore[attr-defined]
        "world/robot",
        rr.Boxes3D(  # type: ignore[attr-defined]
            centers=[[0.0, 0.0, robot.height_m / 2]],
            half_sizes=[[robot.footprint_length_m / 2, robot.footprint_width_m / 2, robot.height_m / 2]],
            colors=[(31, 111, 139, 175)],
            labels=[robot.model],
        ),
        static=True,
    )
    rec.log(  # type: ignore[attr-defined]
        "world/robot/front_axis",
        rr.Arrows3D(vectors=[[0.22, 0.0, 0.0]], origins=[[0.0, 0.0, robot.height_m + 0.04]], colors=[(25, 32, 42, 255)]),  # type: ignore[attr-defined]
        static=True,
    )

    trajectory = [(pose.x, pose.y, 0.055) for pose in replay.poses]
    if len(trajectory) >= 2:
        rec.log(  # type: ignore[attr-defined]
            "world/trajectory",
            rr.LineStrips3D([trajectory], radii=0.018, colors=[(31, 111, 139, 255)]),  # type: ignore[attr-defined]
            static=True,
        )

    estimated_trajectory = [(estimate.x, estimate.y, 0.085) for estimate in replay.odometry_estimates]
    if len(estimated_trajectory) >= 2:
        rec.log(  # type: ignore[attr-defined]
            "world/odometry_estimate_trajectory",
            rr.LineStrips3D([estimated_trajectory], radii=0.012, colors=[(236, 151, 31, 255)]),  # type: ignore[attr-defined]
            static=True,
        )

    if replay.slam_map_points:
        rec.log(  # type: ignore[attr-defined]
            "world/submitted_slam_map",
            rr.Points3D(  # type: ignore[attr-defined]
                [(point.x, point.y, robot.height_m + 0.045) for point in replay.slam_map_points],
                radii=0.014,
                colors=(236, 151, 31, 170),
            ),
            static=True,
        )

    rec.log(  # type: ignore[attr-defined]
        "summary",
        rr.TextDocument(  # type: ignore[attr-defined]
            "\n".join(
                [
                    f"challenge: {challenge.id}",
                    f"status: {metrics.status}",
                    f"score: {metrics.score}",
                    f"robot: {robot.model}",
                ]
            )
        ),
        static=True,
    )


def _log_time_series(*, rr: object, rec: object, challenge: ChallengeSpec, replay: ReplayTrace) -> None:
    if challenge.robot.type == "profiled_cart_1d":
        _log_motion_profile_time_series(rr=rr, rec=rec, challenge=challenge, replay=replay)
        return
    if challenge.robot.type == "cart_pole":
        _log_cart_pole_time_series(rr=rr, rec=rec, challenge=challenge, replay=replay)
        return
    if challenge.robot.type == "manipulator":
        _log_manipulator_time_series(rr=rr, rec=rec, replay=replay)
        return
    if challenge.robot.type == "motor_stand":
        _log_motor_time_series(rr=rr, rec=rec, challenge=challenge, replay=replay)
        return
    if challenge.robot.type == "quadrotor":
        _log_quadrotor_time_series(rr=rr, rec=rec, challenge=challenge, replay=replay)
        return

    robot = challenge.robot
    controls_by_time = {round(control.t, 6): control for control in replay.controls}
    poses_by_time = {round(pose.t, 6): pose for pose in replay.poses}
    odometry_by_time = {round(estimate.t, 6): estimate for estimate in replay.odometry_estimates}

    if robot.cameras:
        camera = robot.cameras[0]
        rec.log(  # type: ignore[attr-defined]
            f"world/robot/{camera.frame}",
            rr.Pinhole(fov_y=math.radians(camera.fov_y_deg), aspect_ratio=camera.width / camera.height),  # type: ignore[attr-defined]
            static=True,
        )

    for index, pose in enumerate(replay.poses):
        rec.set_time("frame", sequence=index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=pose.t)  # type: ignore[attr-defined]
        rec.log(  # type: ignore[attr-defined]
            "world/robot",
            rr.Transform3D(  # type: ignore[attr-defined]
                translation=[pose.x, pose.y, 0.0],
                rotation=rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=pose.yaw),  # type: ignore[attr-defined]
            ),
        )
        control = controls_by_time.get(round(pose.t, 6))
        if control is not None:
            rec.log("controls/left_wheel_velocity", rr.Scalars(control.left_wheel_velocity))  # type: ignore[attr-defined]
            rec.log("controls/right_wheel_velocity", rr.Scalars(control.right_wheel_velocity))  # type: ignore[attr-defined]
        estimate = odometry_by_time.get(round(pose.t, 6))
        if estimate is not None:
            rec.log(  # type: ignore[attr-defined]
                "world/odometry_estimate",
                rr.Transform3D(  # type: ignore[attr-defined]
                    translation=[estimate.x, estimate.y, 0.0],
                    rotation=rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=estimate.yaw),  # type: ignore[attr-defined]
                ),
            )
            rec.log("odometry/position_error_m", rr.Scalars(estimate.position_error_m))  # type: ignore[attr-defined]
            rec.log("odometry/yaw_error_rad", rr.Scalars(estimate.yaw_error_rad))  # type: ignore[attr-defined]

    for scan in replay.lidar_scans:
        pose = poses_by_time.get(round(scan.t, 6)) or _nearest_pose(replay.poses, scan.t)
        points = [
            (
                pose.x + distance * math.cos(pose.yaw + angle),
                pose.y + distance * math.sin(pose.yaw + angle),
                robot.height_m + 0.025,
            )
            for angle, distance in zip(scan.angles_rad, scan.ranges_m, strict=False)
            if distance < scan.max_range_m
        ]
        rec.set_time("frame", sequence=scan.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=scan.t)  # type: ignore[attr-defined]
        rec.log(  # type: ignore[attr-defined]
            "world/lidar_2d_top/points",
            rr.Points3D(points, radii=0.012, colors=(45, 132, 92, 180)),  # type: ignore[attr-defined]
        )

    for sample in replay.encoder_samples:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        rec.log("sensors/encoders/left_ticks", rr.Scalars(sample.left_ticks))  # type: ignore[attr-defined]
        rec.log("sensors/encoders/right_ticks", rr.Scalars(sample.right_ticks))  # type: ignore[attr-defined]

    for sample in replay.gyro_samples:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        rec.log("sensors/gyro/yaw_rate_radps", rr.Scalars(sample.yaw_rate_radps))  # type: ignore[attr-defined]

    for sample in replay.imu_samples:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        rec.log("sensors/imu/ax_mps2", rr.Scalars(sample.ax))  # type: ignore[attr-defined]
        rec.log("sensors/imu/ay_mps2", rr.Scalars(sample.ay))  # type: ignore[attr-defined]
        rec.log("sensors/imu/az_mps2", rr.Scalars(sample.az))  # type: ignore[attr-defined]
        rec.log("sensors/imu/gx_radps", rr.Scalars(sample.gx))  # type: ignore[attr-defined]
        rec.log("sensors/imu/gy_radps", rr.Scalars(sample.gy))  # type: ignore[attr-defined]
        rec.log("sensors/imu/gz_radps", rr.Scalars(sample.gz))  # type: ignore[attr-defined]

    for decision in replay.collision_decisions:
        rec.set_time("frame", sequence=decision.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=decision.t)  # type: ignore[attr-defined]
        rec.log("classifier/contact", rr.Scalars(1.0 if decision.contact else 0.0))  # type: ignore[attr-defined]
        rec.log("classifier/severity", rr.Scalars(_severity_value(decision.severity)))  # type: ignore[attr-defined]

    for sample in replay.charger_pose_samples:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        rec.log("sensors/charger_pose/visible", rr.Scalars(1.0 if sample.visible else 0.0))  # type: ignore[attr-defined]
        if sample.x is None or sample.y is None or sample.yaw is None:
            continue
        rec.log("sensors/charger_pose/x", rr.Scalars(sample.x))  # type: ignore[attr-defined]
        rec.log("sensors/charger_pose/y", rr.Scalars(sample.y))  # type: ignore[attr-defined]
        rec.log("sensors/charger_pose/yaw", rr.Scalars(sample.yaw))  # type: ignore[attr-defined]
        pose = poses_by_time.get(round(sample.t, 6)) or _nearest_pose(replay.poses, sample.t)
        world_x = pose.x + sample.x * math.cos(pose.yaw) - sample.y * math.sin(pose.yaw)
        world_y = pose.y + sample.x * math.sin(pose.yaw) + sample.y * math.cos(pose.yaw)
        rec.log(  # type: ignore[attr-defined]
            "world/charger_pose_estimate",
            rr.Points3D([(world_x, world_y, 0.09)], radii=0.035, colors=(42, 210, 118, 210)),  # type: ignore[attr-defined]
        )

    for sample in replay.april_tag_pose_samples:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        rec.log("sensors/apriltag/tag_id", rr.Scalars(sample.tag_id))  # type: ignore[attr-defined]
        rec.log("sensors/apriltag/distance_m", rr.Scalars(sample.distance_m))  # type: ignore[attr-defined]
        rec.log("sensors/apriltag/bearing_rad", rr.Scalars(sample.bearing_rad))  # type: ignore[attr-defined]
        rec.log("sensors/apriltag/position_std_m", rr.Scalars(sample.position_std_m))  # type: ignore[attr-defined]
        rec.log("sensors/apriltag/yaw_std_rad", rr.Scalars(sample.yaw_std_rad))  # type: ignore[attr-defined]
        rec.log("sensors/apriltag/latency_sec", rr.Scalars(sample.latency_sec))  # type: ignore[attr-defined]
        rec.log("sensors/apriltag/ambiguity", rr.Scalars(sample.ambiguity))  # type: ignore[attr-defined]
        rec.log(  # type: ignore[attr-defined]
            "world/apriltag_pose_measurement",
            rr.Transform3D(  # type: ignore[attr-defined]
                translation=[sample.x, sample.y, 0.0],
                rotation=rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=sample.yaw),  # type: ignore[attr-defined]
            ),
        )

    for frame in replay.render_frames:
        _log_render_frame(rr=rr, rec=rec, frame=frame)


def _log_cart_pole_static_scene(
    *,
    rr: object,
    rec: object,
    challenge: ChallengeSpec,
    replay: ReplayTrace,
    metrics: ScoreMetrics,
) -> None:
    plant = challenge.robot
    track = plant.track_half_width_m
    rec.log(  # type: ignore[attr-defined]
        "world/track",
        rr.LineStrips3D([ [(-track, 0.0, 0.04), (track, 0.0, 0.04)] ], radii=0.02, colors=[(70, 82, 96, 255)]),  # type: ignore[attr-defined]
        static=True,
    )
    rec.log(  # type: ignore[attr-defined]
        "world/track_limits",
        rr.Points3D([(-track, 0.0, 0.12), (track, 0.0, 0.12)], radii=0.05, colors=(184, 62, 54, 220)),  # type: ignore[attr-defined]
        static=True,
    )

    cart_path = [(sample.cart_position_m, 0.0, plant.cart_height_m + 0.07) for sample in replay.cart_pole_states]
    if len(cart_path) >= 2:
        rec.log(  # type: ignore[attr-defined]
            "world/cart_path",
            rr.LineStrips3D([cart_path], radii=0.012, colors=[(31, 111, 139, 255)]),  # type: ignore[attr-defined]
            static=True,
        )

    rec.log(  # type: ignore[attr-defined]
        "summary",
        rr.TextDocument(  # type: ignore[attr-defined]
            "\n".join(
                [
                    f"challenge: {challenge.id}",
                    f"status: {metrics.status}",
                    f"score: {metrics.score}",
                    f"max angle rad: {metrics.max_abs_pole_angle_rad}",
                    f"robot: {plant.model}",
                ]
            )
        ),
        static=True,
    )


def _log_cart_pole_time_series(*, rr: object, rec: object, challenge: ChallengeSpec, replay: ReplayTrace) -> None:
    plant = challenge.robot
    controls_by_time = {round(control.t, 6): control for control in replay.cart_pole_controls}
    cart_z = plant.cart_height_m / 2 + 0.035
    pivot_z = cart_z + plant.cart_height_m / 2 + 0.012
    cart_half_sizes = [[plant.cart_width_m / 2, 0.09, plant.cart_height_m / 2]]

    for sample in replay.cart_pole_states:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        x = sample.cart_position_m
        theta = sample.pole_angle_rad
        pivot = (x, 0.0, pivot_z)
        tip = (
            x + plant.pole_length_m * math.sin(theta),
            0.0,
            pivot_z + plant.pole_length_m * math.cos(theta),
        )
        rec.log(  # type: ignore[attr-defined]
            "world/cart",
            rr.Boxes3D(  # type: ignore[attr-defined]
                centers=[[x, 0.0, cart_z]],
                half_sizes=cart_half_sizes,
                colors=[(31, 111, 139, 190)],
                labels=["cart"],
            ),
        )
        rec.log(  # type: ignore[attr-defined]
            "world/pole",
            rr.LineStrips3D([[pivot, tip]], radii=plant.pole_radius_m, colors=[(224, 143, 46, 255)]),  # type: ignore[attr-defined]
        )
        rec.log("state/cart_position_m", rr.Scalars(sample.cart_position_m))  # type: ignore[attr-defined]
        rec.log("state/cart_velocity_mps", rr.Scalars(sample.cart_velocity_mps))  # type: ignore[attr-defined]
        rec.log("state/pole_angle_rad", rr.Scalars(sample.pole_angle_rad))  # type: ignore[attr-defined]
        rec.log("state/pole_angular_velocity_radps", rr.Scalars(sample.pole_angular_velocity_radps))  # type: ignore[attr-defined]
        if sample.minimum_phase_output_m is not None:
            rec.log("state/minimum_phase_output_m", rr.Scalars(sample.minimum_phase_output_m))  # type: ignore[attr-defined]

        control = controls_by_time.get(round(sample.t, 6))
        if control is not None:
            rec.log("controls/cart_force_n", rr.Scalars(control.force_n))  # type: ignore[attr-defined]

    for frame in replay.render_frames:
        _log_render_frame(rr=rr, rec=rec, frame=frame)


def _log_motion_profile_static_scene(
    *,
    rr: object,
    rec: object,
    challenge: ChallengeSpec,
    replay: ReplayTrace,
    metrics: ScoreMetrics,
) -> None:
    cart = challenge.robot
    track = cart.track_half_width_m
    target_position = float(replay.metadata.get("target_position_m", 0.0))
    rec.log(  # type: ignore[attr-defined]
        "world/track",
        rr.LineStrips3D(  # type: ignore[attr-defined]
            [[(-track, 0.0, 0.04), (track, 0.0, 0.04)]],
            radii=0.02,
            colors=[(70, 82, 96, 255)],
        ),
        static=True,
    )
    rec.log(  # type: ignore[attr-defined]
        "world/track_limits",
        rr.Points3D(  # type: ignore[attr-defined]
            [(-track, 0.0, 0.12), (track, 0.0, 0.12)],
            radii=0.05,
            colors=(184, 62, 54, 220),
        ),
        static=True,
    )
    rec.log(  # type: ignore[attr-defined]
        "world/target",
        rr.Points3D(  # type: ignore[attr-defined]
            [(target_position, 0.0, cart.cart_height_m + 0.12)],
            radii=0.06,
            colors=(210, 52, 44, 255),
        ),
        static=True,
    )

    cart_path = [(sample.position_m, 0.0, cart.cart_height_m + 0.08) for sample in replay.motion_profile_states]
    if len(cart_path) >= 2:
        rec.log(  # type: ignore[attr-defined]
            "world/cart_path",
            rr.LineStrips3D([cart_path], radii=0.012, colors=[(31, 111, 139, 255)]),  # type: ignore[attr-defined]
            static=True,
        )

    rec.log(  # type: ignore[attr-defined]
        "summary",
        rr.TextDocument(  # type: ignore[attr-defined]
            "\n".join(
                [
                    f"challenge: {challenge.id}",
                    f"status: {metrics.status}",
                    f"score: {metrics.score}",
                    f"finish time sec: {metrics.finish_time_sec}",
                    f"optimal time sec: {metrics.optimal_time_sec}",
                    f"robot: {cart.model}",
                ]
            )
        ),
        static=True,
    )


def _log_motion_profile_time_series(*, rr: object, rec: object, challenge: ChallengeSpec, replay: ReplayTrace) -> None:
    cart = challenge.robot
    controls_by_time = {round(control.t, 6): control for control in replay.motion_profile_controls}
    cart_z = cart.cart_height_m / 2 + 0.04
    cart_half_sizes = [[cart.cart_width_m / 2, 0.12, cart.cart_height_m / 2]]

    for sample in replay.motion_profile_states:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        rec.log(  # type: ignore[attr-defined]
            "world/cart",
            rr.Boxes3D(  # type: ignore[attr-defined]
                centers=[[sample.position_m, 0.0, cart_z]],
                half_sizes=cart_half_sizes,
                colors=[(31, 111, 139, 190)],
                labels=["cart"],
            ),
        )
        rec.log("state/position_m", rr.Scalars(sample.position_m))  # type: ignore[attr-defined]
        rec.log("state/velocity_mps", rr.Scalars(sample.velocity_mps))  # type: ignore[attr-defined]
        rec.log("state/acceleration_mps2", rr.Scalars(sample.acceleration_mps2))  # type: ignore[attr-defined]
        rec.log("state/position_error_m", rr.Scalars(sample.position_error_m))  # type: ignore[attr-defined]
        rec.log("state/velocity_error_mps", rr.Scalars(sample.velocity_error_mps))  # type: ignore[attr-defined]
        rec.log("target/position_m", rr.Scalars(sample.target_position_m))  # type: ignore[attr-defined]

        control = controls_by_time.get(round(sample.t, 6))
        if control is not None:
            rec.log("controls/acceleration_command_mps2", rr.Scalars(control.acceleration_command_mps2))  # type: ignore[attr-defined]
            rec.log("controls/applied_acceleration_mps2", rr.Scalars(control.applied_acceleration_mps2))  # type: ignore[attr-defined]
            rec.log("limits/acceleration_violation", rr.Scalars(1.0 if control.acceleration_limit_violation else 0.0))  # type: ignore[attr-defined]
        rec.log("limits/velocity_violation", rr.Scalars(1.0 if sample.velocity_limit_violation else 0.0))  # type: ignore[attr-defined]

    for frame in replay.render_frames:
        _log_render_frame(rr=rr, rec=rec, frame=frame)


def _log_quadrotor_static_scene(
    *,
    rr: object,
    rec: object,
    challenge: ChallengeSpec,
    replay: ReplayTrace,
    metrics: ScoreMetrics,
) -> None:
    robot = challenge.robot
    bounds = replay.metadata.get("flight_volume_m", {})
    gates = replay.metadata.get("gates", [])

    rec.log(  # type: ignore[attr-defined]
        "world/flight_volume",
        rr.LineStrips3D(_flight_volume_strips(bounds), radii=0.01, colors=[(92, 105, 125, 150)]),  # type: ignore[attr-defined]
        static=True,
    )

    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            gate_id = str(gate.get("id", "gate"))
            rec.log(  # type: ignore[attr-defined]
                f"world/gates/{gate_id}",
                rr.LineStrips3D(  # type: ignore[attr-defined]
                    [_gate_rectangle(gate)],
                    radii=0.025,
                    colors=[(232, 138, 35, 255)],
                ),
                static=True,
            )
            try:
                yaw = float(gate.get("yaw", 0.0))
                x = float(gate["x"])
                y = float(gate["y"])
                z = float(gate["z"])
            except (KeyError, TypeError, ValueError):
                continue
            rec.log(  # type: ignore[attr-defined]
                f"world/gates/{gate_id}/normal",
                rr.Arrows3D(  # type: ignore[attr-defined]
                    vectors=[[0.24 * math.cos(yaw), 0.24 * math.sin(yaw), 0.0]],
                    origins=[[x, y, z]],
                    colors=[(232, 138, 35, 255)],
                ),
                static=True,
            )

    rec.log(  # type: ignore[attr-defined]
        "world/drone",
        rr.Boxes3D(  # type: ignore[attr-defined]
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[[robot.footprint_length_m / 2, robot.footprint_width_m / 2, robot.height_m / 2]],
            colors=[(31, 111, 139, 155)],
            labels=[robot.model],
        ),
        static=True,
    )

    trajectory = [(pose.x, pose.y, float(pose.z or 0.0)) for pose in replay.poses]
    if len(trajectory) >= 2:
        rec.log(  # type: ignore[attr-defined]
            "world/trajectory",
            rr.LineStrips3D([trajectory], radii=0.014, colors=[(31, 111, 139, 255)]),  # type: ignore[attr-defined]
            static=True,
        )

    rec.log(  # type: ignore[attr-defined]
        "summary",
        rr.TextDocument(  # type: ignore[attr-defined]
            "\n".join(
                [
                    f"challenge: {challenge.id}",
                    f"status: {metrics.status}",
                    f"score: {metrics.score}",
                    f"gates: {metrics.gates_completed}/{metrics.gate_count}",
                    f"robot: {robot.model}",
                ]
            )
        ),
        static=True,
    )


def _log_quadrotor_time_series(*, rr: object, rec: object, challenge: ChallengeSpec, replay: ReplayTrace) -> None:
    robot = challenge.robot
    controls_by_time = {round(control.t, 6): control for control in replay.controls}

    if robot.cameras:
        camera = robot.cameras[0]
        rec.log(  # type: ignore[attr-defined]
            f"world/drone/{camera.frame}",
            rr.Pinhole(fov_y=math.radians(camera.fov_y_deg), aspect_ratio=camera.width / camera.height),  # type: ignore[attr-defined]
            static=True,
        )

    for index, pose in enumerate(replay.poses):
        rec.set_time("frame", sequence=index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=pose.t)  # type: ignore[attr-defined]
        rec.log(  # type: ignore[attr-defined]
            "world/drone",
            rr.Transform3D(  # type: ignore[attr-defined]
                translation=[pose.x, pose.y, float(pose.z or 0.0)],
                rotation=rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=pose.yaw),  # type: ignore[attr-defined]
            ),
        )
        rec.log("state/altitude_m", rr.Scalars(float(pose.z or 0.0)))  # type: ignore[attr-defined]
        rec.log("state/roll_rad", rr.Scalars(float(pose.roll or 0.0)))  # type: ignore[attr-defined]
        rec.log("state/pitch_rad", rr.Scalars(float(pose.pitch or 0.0)))  # type: ignore[attr-defined]
        rec.log("state/yaw_rad", rr.Scalars(pose.yaw))  # type: ignore[attr-defined]
        rec.log("state/speed_mps", rr.Scalars(float(pose.speed or 0.0)))  # type: ignore[attr-defined]
        control = controls_by_time.get(round(pose.t, 6))
        if control is not None:
            rec.log("controls/roll_rate_radps", rr.Scalars(float(control.roll_rate_radps or 0.0)))  # type: ignore[attr-defined]
            rec.log("controls/pitch_rate_radps", rr.Scalars(float(control.pitch_rate_radps or 0.0)))  # type: ignore[attr-defined]
            rec.log("controls/yaw_rate_radps", rr.Scalars(float(control.yaw_rate_radps or 0.0)))  # type: ignore[attr-defined]
            rec.log("controls/collective_power", rr.Scalars(float(control.power or 0.0)))  # type: ignore[attr-defined]

    for frame in replay.render_frames:
        _log_render_frame(rr=rr, rec=rec, frame=frame)


def _log_manipulator_static_scene(
    *,
    rr: object,
    rec: object,
    challenge: ChallengeSpec,
    replay: ReplayTrace,
    metrics: ScoreMetrics,
) -> None:
    robot = challenge.robot
    target_path = _target_sequence_strip(replay.metadata.get("target_sequence"))
    if len(target_path) >= 2:
        rec.log(  # type: ignore[attr-defined]
            "world/target_path",
            rr.LineStrips3D([target_path], radii=0.008, colors=[(206, 58, 48, 210)]),  # type: ignore[attr-defined]
            static=True,
        )
    actual_path = [(sample.actual_x, sample.actual_y, sample.actual_z + 0.045) for sample in replay.inverse_kinematics_samples]
    if len(actual_path) >= 2:
        rec.log(  # type: ignore[attr-defined]
            "world/end_effector_path",
            rr.LineStrips3D([actual_path], radii=0.010, colors=[(31, 111, 139, 255)]),  # type: ignore[attr-defined]
            static=True,
        )
    if getattr(robot, "mechanism", "") == "five_bar_scara":
        spacing = float(robot.base_spacing_m or 0.36)
        rec.log(  # type: ignore[attr-defined]
            "world/base_anchors",
            rr.Points3D([(-spacing / 2.0, 0.0, 0.045), (spacing / 2.0, 0.0, 0.045)], radii=0.04, colors=(18, 24, 32, 255)),  # type: ignore[attr-defined]
            static=True,
        )
    else:
        rec.log(  # type: ignore[attr-defined]
            "world/base",
            rr.Points3D([(0.0, 0.0, float(getattr(robot, "base_height_m", 0.0)))], radii=0.045, colors=(18, 24, 32, 255)),  # type: ignore[attr-defined]
            static=True,
        )
    rec.log(  # type: ignore[attr-defined]
        "summary",
        rr.TextDocument(  # type: ignore[attr-defined]
            "\n".join(
                [
                    f"challenge: {challenge.id}",
                    f"status: {metrics.status}",
                    f"score: {metrics.score}",
                    f"mechanism: {getattr(robot, 'mechanism', 'manipulator')}",
                    f"mean error m: {metrics.mean_position_error_m}",
                    f"max error m: {metrics.max_position_error_m}",
                ]
            )
        ),
        static=True,
    )


def _log_manipulator_time_series(*, rr: object, rec: object, replay: ReplayTrace) -> None:
    for sample in replay.inverse_kinematics_samples:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        rec.log(  # type: ignore[attr-defined]
            "world/current_target",
            rr.Points3D([(sample.target_x, sample.target_y, sample.target_z + 0.045)], radii=0.026, colors=(210, 52, 44, 255)),  # type: ignore[attr-defined]
        )
        rec.log(  # type: ignore[attr-defined]
            "world/end_effector",
            rr.Points3D([(sample.actual_x, sample.actual_y, sample.actual_z + 0.045)], radii=0.024, colors=(31, 111, 139, 255)),  # type: ignore[attr-defined]
        )
        rec.log("ik/position_error_m", rr.Scalars(sample.position_error_m))  # type: ignore[attr-defined]
        rec.log("ik/joint_limit_violation", rr.Scalars(1.0 if sample.joint_limit_violation else 0.0))  # type: ignore[attr-defined]
        for index, angle in enumerate(sample.joint_angles_rad):
            rec.log(f"ik/joint_{index}_rad", rr.Scalars(angle))  # type: ignore[attr-defined]

    for frame in replay.render_frames:
        _log_render_frame(rr=rr, rec=rec, frame=frame)


def _log_motor_static_scene(
    *,
    rr: object,
    rec: object,
    challenge: ChallengeSpec,
    replay: ReplayTrace,
    metrics: ScoreMetrics,
) -> None:
    fixture = challenge.robot
    base_z = fixture.base_height_m / 2

    rec.log(  # type: ignore[attr-defined]
        "world/fixture/base",
        rr.Boxes3D(  # type: ignore[attr-defined]
            centers=[[0.0, 0.0, base_z]],
            half_sizes=[[fixture.base_length_m / 2, fixture.base_width_m / 2, base_z]],
            colors=[(42, 55, 64, 180)],
            labels=[fixture.model],
        ),
        static=True,
    )
    rec.log(  # type: ignore[attr-defined]
        "world/fixture/scale",
        rr.Boxes3D(  # type: ignore[attr-defined]
            centers=[[0.085, 0.0, 0.068]],
            half_sizes=[[fixture.scale_plate_width_m / 2, fixture.scale_plate_depth_m / 2, 0.008]],
            colors=[(82, 111, 127, 180)],
            labels=["scale plate"],
        ),
        static=True,
    )
    rec.log(  # type: ignore[attr-defined]
        "summary",
        rr.TextDocument(  # type: ignore[attr-defined]
            "\n".join(
                [
                    f"challenge: {challenge.id}",
                    f"status: {metrics.status}",
                    f"score: {metrics.score}",
                    f"target_force_n: {metrics.target_force_n}",
                    f"fixture: {fixture.model}",
                ]
            )
        ),
        static=True,
    )


def _log_motor_time_series(*, rr: object, rec: object, challenge: ChallengeSpec, replay: ReplayTrace) -> None:
    fixture = challenge.robot
    hinge_x = -0.075
    hinge_z = 0.175
    press_foot_length = 0.086

    for sample in replay.motor_states:
        rec.set_time("frame", sequence=sample.frame_index)  # type: ignore[attr-defined]
        rec.set_time("time", duration=sample.t)  # type: ignore[attr-defined]
        tip_x = hinge_x + fixture.shaft_length_m * math.cos(sample.shaft_angle_rad)
        tip_z = hinge_z - fixture.shaft_length_m * math.sin(sample.shaft_angle_rad)
        rec.log(  # type: ignore[attr-defined]
            "world/fixture/shaft",
            rr.LineStrips3D(  # type: ignore[attr-defined]
                [
                    [(hinge_x, 0.0, hinge_z), (tip_x, 0.0, tip_z)],
                    [(tip_x, 0.0, tip_z), (tip_x, 0.0, tip_z - press_foot_length)],
                ],
                radii=fixture.shaft_radius_m,
                colors=[(194, 199, 202, 255)],
            ),
        )
        rec.log("sensors/scale_force_n", rr.Scalars(sample.scale_force_n))  # type: ignore[attr-defined]
        rec.log("sensors/measured_current_a", rr.Scalars(sample.measured_current_a))  # type: ignore[attr-defined]
        rec.log("sensors/force_error_n", rr.Scalars(sample.force_error_n))  # type: ignore[attr-defined]
        rec.log("state/shaft_angle_rad", rr.Scalars(sample.shaft_angle_rad))  # type: ignore[attr-defined]
        rec.log("state/motor_torque_nm", rr.Scalars(sample.motor_torque_nm))  # type: ignore[attr-defined]
        rec.log("target/force_n", rr.Scalars(sample.target_force_n))  # type: ignore[attr-defined]

    for control in replay.motor_controls:
        rec.set_time("time", duration=control.t)  # type: ignore[attr-defined]
        rec.log("controls/current_command_a", rr.Scalars(control.current_command_a))  # type: ignore[attr-defined]

    for frame in replay.render_frames:
        _log_render_frame(rr=rr, rec=rec, frame=frame)


def _log_render_frame(*, rr: object, rec: object, frame: RenderFrame) -> None:
    image_bytes = _decode_data_url(frame.image_data_url)
    if image_bytes is None:
        return
    rec.set_time("frame", sequence=frame.frame_index)  # type: ignore[attr-defined]
    rec.set_time("time", duration=frame.t)  # type: ignore[attr-defined]
    rec.log(  # type: ignore[attr-defined]
        f"mujoco/{frame.camera}",
        rr.EncodedImage(contents=image_bytes, media_type="image/png"),  # type: ignore[attr-defined]
    )


def _bounds_strip(bounds: object) -> list[tuple[float, float, float]]:
    if not isinstance(bounds, dict):
        return []
    x_min = float(bounds.get("x_min", -2.5))
    x_max = float(bounds.get("x_max", 2.5))
    y_min = float(bounds.get("y_min", -2.0))
    y_max = float(bounds.get("y_max", 2.0))
    return [
        (x_min, y_min, 0.02),
        (x_max, y_min, 0.02),
        (x_max, y_max, 0.02),
        (x_min, y_max, 0.02),
        (x_min, y_min, 0.02),
    ]


def _flight_volume_strips(bounds: object) -> list[list[tuple[float, float, float]]]:
    if not isinstance(bounds, dict):
        return []
    x_min = float(bounds.get("x_min", -0.5))
    x_max = float(bounds.get("x_max", 4.5))
    y_min = float(bounds.get("y_min", -1.5))
    y_max = float(bounds.get("y_max", 1.5))
    z_min = float(bounds.get("z_min", 0.15))
    z_max = float(bounds.get("z_max", 2.25))
    bottom = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_min),
    ]
    top = [
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
        (x_min, y_min, z_max),
    ]
    verticals = [
        [(x_min, y_min, z_min), (x_min, y_min, z_max)],
        [(x_max, y_min, z_min), (x_max, y_min, z_max)],
        [(x_max, y_max, z_min), (x_max, y_max, z_max)],
        [(x_min, y_max, z_min), (x_min, y_max, z_max)],
    ]
    return [bottom, top, *verticals]


def _gate_rectangle(gate: dict[str, object]) -> list[tuple[float, float, float]]:
    try:
        x = float(gate["x"])
        y = float(gate["y"])
        z = float(gate["z"])
        yaw = float(gate.get("yaw", 0.0))
        width = float(gate["width_m"])
        height = float(gate["height_m"])
    except (KeyError, TypeError, ValueError):
        return []

    lateral = (-math.sin(yaw), math.cos(yaw), 0.0)
    half_width = width / 2.0
    half_height = height / 2.0
    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
        (-half_width, -half_height),
    ]
    return [
        (
            x + lateral_offset * lateral[0],
            y + lateral_offset * lateral[1],
            z + vertical_offset,
        )
        for lateral_offset, vertical_offset in corners
    ]


def _circle_strip(cx: float, cy: float, radius: float, *, z: float, segments: int = 72) -> list[tuple[float, float, float]]:
    return [
        (
            cx + radius * math.cos((2.0 * math.pi * index) / segments),
            cy + radius * math.sin((2.0 * math.pi * index) / segments),
            z,
        )
        for index in range(segments + 1)
    ]


def _route_strip(route_waypoints: object) -> list[tuple[float, float, float]]:
    if not isinstance(route_waypoints, list):
        return []

    points: list[tuple[float, float, float]] = []
    for waypoint in route_waypoints:
        if not isinstance(waypoint, dict):
            continue
        try:
            points.append((float(waypoint["x"]), float(waypoint["y"]), 0.06))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _target_sequence_strip(target_sequence: object) -> list[tuple[float, float, float]]:
    if not isinstance(target_sequence, list):
        return []

    points: list[tuple[float, float, float]] = []
    for target in target_sequence:
        if not isinstance(target, dict):
            continue
        try:
            points.append((float(target["x"]), float(target["y"]), float(target.get("z", 0.0)) + 0.045))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _tag_layout_points(tag_layout: object, *, z: float) -> list[tuple[float, float, float]]:
    if not isinstance(tag_layout, list):
        return []

    points: list[tuple[float, float, float]] = []
    for tag in tag_layout:
        if not isinstance(tag, dict):
            continue
        try:
            points.append((float(tag["x"]), float(tag["y"]), z))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _nearest_pose(poses: Iterable[PoseSample], t: float) -> PoseSample:
    return min(poses, key=lambda pose: abs(pose.t - t))


def _decode_data_url(data_url: str) -> bytes | None:
    if not data_url.startswith("data:image/") or "," not in data_url:
        return None
    _, encoded = data_url.split(",", 1)
    try:
        return base64.b64decode(encoded)
    except ValueError:
        return None


def _severity_value(severity: str) -> float:
    return {"none": 0.0, "light": 1.0, "hard": 2.0}.get(severity, 0.0)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return slug or "run"
