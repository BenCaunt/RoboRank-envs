from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from roborank_envs.catalog import INVERSE_KINEMATICS_1
from roborank_envs.models import ChallengeSpec, InverseKinematicsSample, PoseSample, RenderFrame, ReplayTrace, RunResult, ScoreMetrics
from roborank_envs.policy_api import IKTarget, InverseKinematicsTask as InverseKinematicsApi
from roborank_envs.policy_api import RobotPolicyProtocol
from roborank_envs.simulation.manipulator_world import ManipulatorMujocoWorld, TargetVisual
from roborank_envs.simulation.rendering import configured_render_interval_sec
from roborank_envs.simulation.rerun_export import write_rerun_recording


Vector3 = tuple[float, float, float]
Segment = tuple[Vector3, Vector3]


@dataclass(frozen=True)
class KinematicTarget:
    index: int
    x: float
    y: float
    z: float
    label: str


class SimulationError(RuntimeError):
    pass


class InverseKinematicsRunner:
    def __init__(self, challenge: ChallengeSpec = INVERSE_KINEMATICS_1) -> None:
        self.challenge = challenge
        self.dt = float(challenge.defaults["dt_sec"])
        self.default_max_steps = int(challenge.defaults["max_steps"])
        self.render_interval_sec = configured_render_interval_sec(
            float(challenge.defaults.get("render_interval_sec", 0.12))
        )
        if challenge.robot.type != "manipulator":
            raise ValueError("InverseKinematicsRunner requires a manipulator challenge.")

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        robot = self.challenge.robot
        max_step_count = max_steps or self.default_max_steps
        targets = self._scenario(seed)[:max_step_count]
        tolerance = float(self.challenge.success_conditions["max_position_error_m"])
        policy_arm = InverseKinematicsApi(
            mechanism=robot.mechanism,
            joint_count=robot.joint_count,
            link_lengths_m=robot.link_lengths_m,
            joint_limits_rad=robot.joint_limits_rad,
            base_height_m=robot.base_height_m,
            base_spacing_m=robot.base_spacing_m,
            tolerance_m=tolerance,
            dt=self.dt,
            max_steps=len(targets),
            seed=seed,
        )
        world = ManipulatorMujocoWorld.create(
            robot=robot,
            targets=[TargetVisual(x=target.x, y=target.y, z=_visual_z(target.z)) for target in targets],
        )

        samples: list[InverseKinematicsSample] = []
        poses: list[PoseSample] = []
        render_frames: list[RenderFrame] = []
        render_error: str | None = None
        status = "accuracy_error"

        try:
            render_stride = max(1, round(self.render_interval_sec / self.dt))
            for index, target in enumerate(targets):
                t = round(index * self.dt, 6)
                policy_arm._update(
                    time=t,
                    target=IKTarget(
                        index=target.index,
                        x=target.x,
                        y=target.y,
                        z=target.z,
                        tolerance_m=tolerance,
                        label=target.label,
                    ),
                    target_count=len(targets),
                )
                joint_angles = self._call_policy_step(policy, policy_arm)
                actual, segments, valid = forward_kinematics(
                    mechanism=robot.mechanism,
                    link_lengths=tuple(robot.link_lengths_m),
                    joint_angles=joint_angles,
                    base_height=robot.base_height_m,
                    base_spacing=robot.base_spacing_m,
                )
                violation = _joint_limit_violation(joint_angles, robot.joint_limits_rad)
                error = _distance(actual, (target.x, target.y, target.z)) if valid else float("inf")
                safe_error = error if math.isfinite(error) else 1e6
                samples.append(
                    InverseKinematicsSample(
                        t=t,
                        frame_index=index,
                        target_x=round(target.x, 6),
                        target_y=round(target.y, 6),
                        target_z=round(target.z, 6),
                        actual_x=round(actual[0], 6) if math.isfinite(actual[0]) else 0.0,
                        actual_y=round(actual[1], 6) if math.isfinite(actual[1]) else 0.0,
                        actual_z=round(actual[2], 6) if math.isfinite(actual[2]) else 0.0,
                        position_error_m=round(safe_error, 6),
                        joint_angles_rad=[round(angle, 6) for angle in joint_angles],
                        joint_limit_violation=violation,
                    )
                )
                poses.append(
                    PoseSample(
                        t=t,
                        x=round(actual[0], 6) if math.isfinite(actual[0]) else 0.0,
                        y=round(actual[1], 6) if math.isfinite(actual[1]) else 0.0,
                        z=round(actual[2], 6) if math.isfinite(actual[2]) else 0.0,
                        yaw=0.0,
                        speed=round(safe_error, 6),
                    )
                )

                world.update(
                    segments=segments,
                    target=(target.x, target.y, _visual_z(target.z)),
                )
                if index == 0 or index == len(targets) - 1 or index % render_stride == 0:
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t,
                        frame_index=index,
                        render_error=render_error,
                    )
        finally:
            world.close()

        metrics = self._score(samples=samples)
        status = metrics.status
        replay = ReplayTrace(
            poses=poses,
            inverse_kinematics_samples=samples,
            render_frames=render_frames,
            metadata={
                "dt_sec": self.dt,
                "max_steps": len(targets),
                "seed": seed,
                "runner": "inverse_kinematics",
                "challenge_mode": self.challenge.id,
                "mechanism": robot.mechanism,
                "target_shape": self.challenge.defaults.get("target_shape", "seeded_points"),
                "target_sequence": [
                    {"index": target.index, "x": target.x, "y": target.y, "z": target.z, "label": target.label}
                    for target in targets
                ],
                "joint_limits_rad": robot.joint_limits_rad,
                "link_lengths_m": robot.link_lengths_m,
                "base_height_m": robot.base_height_m,
                "base_spacing_m": robot.base_spacing_m,
                "mujoco_backend": world.backend_name,
                "simulation_mode": "analytic_inverse_kinematics_with_mujoco_render"
                if world.available
                else "analytic_inverse_kinematics_fallback",
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
                "status": status,
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

    def _scenario(self, seed: int) -> list[KinematicTarget]:
        mechanism = self.challenge.robot.mechanism
        count = self.default_max_steps
        rng = random.Random(seed)
        if mechanism == "planar_2link":
            return self._planar_targets(rng=rng, count=count)
        if mechanism == "turret_2link":
            return self._turret_targets(rng=rng, count=count)
        if mechanism == "five_bar_scara":
            return self._five_bar_targets(count=count)
        raise SimulationError(f"Unsupported IK mechanism: {mechanism}")

    def _planar_targets(self, *, rng: random.Random, count: int) -> list[KinematicTarget]:
        targets: list[KinematicTarget] = []
        for index in range(count):
            shoulder = rng.uniform(-2.35, 2.35)
            elbow = rng.uniform(0.35, 2.35)
            point, _, _ = forward_kinematics(
                mechanism="planar_2link",
                link_lengths=tuple(self.challenge.robot.link_lengths_m),
                joint_angles=(shoulder, elbow),
                base_height=0.0,
                base_spacing=None,
            )
            targets.append(KinematicTarget(index=index, x=point[0], y=point[1], z=0.0, label=f"point_{index}"))
        return targets

    def _turret_targets(self, *, rng: random.Random, count: int) -> list[KinematicTarget]:
        targets: list[KinematicTarget] = []
        for index in range(count):
            yaw = rng.uniform(-2.55, 2.55)
            shoulder = rng.uniform(-0.18, 0.82)
            elbow = rng.uniform(-1.30, 0.92)
            point, _, _ = forward_kinematics(
                mechanism="turret_2link",
                link_lengths=tuple(self.challenge.robot.link_lengths_m),
                joint_angles=(yaw, shoulder, elbow),
                base_height=self.challenge.robot.base_height_m,
                base_spacing=None,
            )
            targets.append(KinematicTarget(index=index, x=point[0], y=point[1], z=point[2], label=f"point_{index}"))
        return targets

    def _five_bar_targets(self, *, count: int) -> list[KinematicTarget]:
        targets: list[KinematicTarget] = []
        center_y = 0.405
        for index in range(count):
            phase = (2.0 * math.pi * index) / max(1, count)
            x = 0.115 * math.sin(phase)
            y = center_y + 0.070 * math.sin(2.0 * phase) * math.cos(phase / 2.0)
            if index > count * 0.55:
                y += 0.028 * math.sin(3.0 * phase)
            targets.append(KinematicTarget(index=index, x=x, y=y, z=0.0, label=f"stroke_{index}"))
        return targets

    def _call_policy_step(self, policy: RobotPolicyProtocol, arm: InverseKinematicsApi) -> tuple[float, ...]:
        try:
            arm._clear_submission()
            step_result = policy.step(arm)
            if step_result is not None:
                raise SimulationError("RobotPolicy.step(arm) should call arm.submit_joint_angles([...]), not return an action.")
            return arm._consume_joint_angles()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _score(self, *, samples: list[InverseKinematicsSample]) -> ScoreMetrics:
        if not samples:
            return ScoreMetrics(
                metric_kind="inverse_kinematics",
                score=0.0,
                success=False,
                status="error",
                elapsed_sec=0.0,
                distance_to_target_m=0.0,
                collision_count=0,
                path_length_m=0.0,
                energy_used=0.0,
                smoothness_cost=0.0,
                target_count=0,
                joint_limit_violation_count=0,
            )

        errors = [sample.position_error_m for sample in samples]
        max_error = max(errors)
        mean_error = sum(errors) / len(errors)
        final_error = errors[-1]
        violation_count = sum(1 for sample in samples if sample.joint_limit_violation)
        max_tolerance = float(self.challenge.success_conditions["max_position_error_m"])
        mean_tolerance = float(self.challenge.success_conditions["mean_position_error_m"])
        success = max_error <= max_tolerance and mean_error <= mean_tolerance and violation_count == 0
        status = "success" if success else "limit_violation" if violation_count else "accuracy_error"
        path_length = _path_length([(sample.actual_x, sample.actual_y, sample.actual_z) for sample in samples])
        smoothness = _joint_smoothness(samples)
        effort = sum(sum(abs(angle) for angle in sample.joint_angles_rad) for sample in samples) * self.dt
        accuracy_score = 75.0 * max(0.0, 1.0 - mean_error / max(mean_tolerance * 4.0, 1e-6))
        worst_score = 20.0 * max(0.0, 1.0 - max_error / max(max_tolerance * 4.0, 1e-6))
        limit_score = 5.0 * max(0.0, 1.0 - violation_count / max(1, len(samples)))
        score = max(0.0, min(100.0, accuracy_score + worst_score + limit_score - min(10.0, smoothness * 0.08)))
        return ScoreMetrics(
            metric_kind="inverse_kinematics",
            score=round(score, 2),
            success=success,
            status=status,
            elapsed_sec=round(samples[-1].t + self.dt, 6),
            distance_to_target_m=round(max_error, 6),
            heading_error_rad=None,
            collision_count=0,
            path_length_m=round(path_length, 6),
            energy_used=round(effort, 6),
            smoothness_cost=round(smoothness, 6),
            final_position_error_m=round(final_error, 6),
            mean_position_error_m=round(mean_error, 6),
            max_position_error_m=round(max_error, 6),
            target_count=len(samples),
            joint_limit_violation_count=violation_count,
        )

    def _append_render_frames(
        self,
        *,
        world: ManipulatorMujocoWorld,
        frames: list[RenderFrame],
        t: float,
        frame_index: int,
        render_error: str | None,
    ) -> str | None:
        if not world.available:
            return render_error
        next_error = render_error
        for camera in world.render_cameras:
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
            except Exception as exc:  # noqa: BLE001 - rendering should not invalidate grading.
                next_error = next_error or f"{type(exc).__name__}: {exc}"
        return next_error


def forward_kinematics(
    *,
    mechanism: str,
    link_lengths: tuple[float, ...],
    joint_angles: tuple[float, ...],
    base_height: float,
    base_spacing: float | None,
) -> tuple[Vector3, list[Segment], bool]:
    if mechanism == "planar_2link":
        return _planar_fk(link_lengths=link_lengths, joint_angles=joint_angles)
    if mechanism == "turret_2link":
        return _turret_fk(link_lengths=link_lengths, joint_angles=joint_angles, base_height=base_height)
    if mechanism == "five_bar_scara":
        return _five_bar_fk(link_lengths=link_lengths, joint_angles=joint_angles, base_spacing=float(base_spacing or 0.36))
    return (float("nan"), float("nan"), float("nan")), [], False


def _planar_fk(*, link_lengths: tuple[float, ...], joint_angles: tuple[float, ...]) -> tuple[Vector3, list[Segment], bool]:
    l1, l2 = link_lengths[:2]
    q0, q1 = joint_angles[:2]
    base = (0.0, 0.0, 0.045)
    elbow = (l1 * math.cos(q0), l1 * math.sin(q0), 0.045)
    tool = (
        elbow[0] + l2 * math.cos(q0 + q1),
        elbow[1] + l2 * math.sin(q0 + q1),
        0.045,
    )
    return (tool[0], tool[1], 0.0), [(base, elbow), (elbow, tool)], True


def _turret_fk(
    *,
    link_lengths: tuple[float, ...],
    joint_angles: tuple[float, ...],
    base_height: float,
) -> tuple[Vector3, list[Segment], bool]:
    l1, l2 = link_lengths[:2]
    yaw, shoulder, elbow = joint_angles[:3]
    pivot = (0.0, 0.0, base_height)
    elbow_radius = l1 * math.cos(shoulder)
    elbow_z = base_height + l1 * math.sin(shoulder)
    tool_radius = elbow_radius + l2 * math.cos(shoulder + elbow)
    tool_z = elbow_z + l2 * math.sin(shoulder + elbow)
    elbow_point = (elbow_radius * math.cos(yaw), elbow_radius * math.sin(yaw), elbow_z)
    tool = (tool_radius * math.cos(yaw), tool_radius * math.sin(yaw), tool_z)
    return tool, [(pivot, elbow_point), (elbow_point, tool)], True


def _five_bar_fk(
    *,
    link_lengths: tuple[float, ...],
    joint_angles: tuple[float, ...],
    base_spacing: float,
) -> tuple[Vector3, list[Segment], bool]:
    l1, l2 = link_lengths[:2]
    q_left, q_right = joint_angles[:2]
    left_anchor = (-base_spacing / 2.0, 0.0, 0.045)
    right_anchor = (base_spacing / 2.0, 0.0, 0.045)
    left_elbow = (
        left_anchor[0] + l1 * math.cos(q_left),
        left_anchor[1] + l1 * math.sin(q_left),
        0.045,
    )
    right_elbow = (
        right_anchor[0] + l1 * math.cos(q_right),
        right_anchor[1] + l1 * math.sin(q_right),
        0.045,
    )
    intersection = _circle_intersection_high_y(left_elbow, right_elbow, l2)
    if intersection is None:
        return (
            (float("nan"), float("nan"), float("nan")),
            [(left_anchor, left_elbow), (right_anchor, right_elbow)],
            False,
        )
    tool = (intersection[0], intersection[1], 0.045)
    return (
        (tool[0], tool[1], 0.0),
        [(left_anchor, left_elbow), (left_elbow, tool), (right_anchor, right_elbow), (right_elbow, tool)],
        True,
    )


def _circle_intersection_high_y(left: Vector3, right: Vector3, radius: float) -> tuple[float, float] | None:
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9 or distance > 2.0 * radius:
        return None
    midpoint = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    half_distance = distance / 2.0
    height_sq = radius**2 - half_distance**2
    if height_sq < -1e-9:
        return None
    height = math.sqrt(max(0.0, height_sq))
    perp = (-dy / distance, dx / distance)
    candidate_a = (midpoint[0] + perp[0] * height, midpoint[1] + perp[1] * height)
    candidate_b = (midpoint[0] - perp[0] * height, midpoint[1] - perp[1] * height)
    return candidate_a if candidate_a[1] >= candidate_b[1] else candidate_b


def _joint_limit_violation(joint_angles: tuple[float, ...], limits: list[list[float]]) -> bool:
    for angle, limit in zip(joint_angles, limits, strict=False):
        lower, upper = float(limit[0]), float(limit[1])
        if angle < lower - 1e-6 or angle > upper + 1e-6:
            return True
    return False


def _distance(actual: Vector3, target: Vector3) -> float:
    return math.sqrt((actual[0] - target[0]) ** 2 + (actual[1] - target[1]) ** 2 + (actual[2] - target[2]) ** 2)


def _path_length(points: list[Vector3]) -> float:
    total = 0.0
    for previous, current in zip(points, points[1:], strict=False):
        total += _distance(previous, current)
    return total


def _joint_smoothness(samples: list[InverseKinematicsSample]) -> float:
    total = 0.0
    for previous, current in zip(samples, samples[1:], strict=False):
        for left, right in zip(previous.joint_angles_rad, current.joint_angles_rad, strict=False):
            total += (right - left) ** 2
    return total


def _visual_z(z: float) -> float:
    return z if abs(z) > 1e-9 else 0.045
