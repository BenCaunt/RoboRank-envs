from __future__ import annotations

import math
import random
from dataclasses import dataclass

from roborank_envs.catalog import IMU_COLLISION_DETECTION
from roborank_envs.models import (
    ChallengeSpec,
    CollisionDecisionSample,
    CollisionEvent,
    ControlSample,
    ImuTraceSample,
    Obstacle,
    Pose2D,
    PoseSample,
    RenderFrame,
    ReplayTrace,
    RunResult,
    ScoreMetrics,
    Target,
)
from roborank_envs.policy_api import CollisionProbe, ImuSample, RobotPolicyProtocol
from roborank_envs.simulation.mujoco_world import MujocoWorld
from roborank_envs.simulation.rendering import configured_render_interval_sec
from roborank_envs.simulation.rerun_export import write_rerun_recording

DEFAULT_REPLAY_RENDER_INTERVAL_SEC = 0.1


@dataclass(frozen=True)
class ArenaBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class ImuCollisionScenario:
    start: Pose2D
    target: Target
    obstacles: list[Obstacle]
    bounds: ArenaBounds


@dataclass(frozen=True)
class DetectionStats:
    actual_collision_time_sec: float | None
    detected: bool
    detection_latency_sec: float | None
    false_positive_count: int
    first_detection_time_sec: float | None


class SimulationError(RuntimeError):
    pass


class ImuCollisionRunner:
    """Classifier challenge runner that drives a hidden mobile probe into one contact."""

    def __init__(self, challenge: ChallengeSpec = IMU_COLLISION_DETECTION) -> None:
        self.challenge = challenge
        self.dt = float(challenge.defaults["dt_sec"])
        self.default_max_steps = int(challenge.defaults["max_steps"])
        self.render_interval_sec = configured_render_interval_sec(
            float(challenge.defaults.get("render_interval_sec", DEFAULT_REPLAY_RENDER_INTERVAL_SEC))
        )
        self.latency_limit_sec = float(challenge.success_conditions["latency_sec"])
        self.post_collision_steps = int(float(challenge.defaults["post_collision_sec"]) / self.dt)
        bounds = challenge.defaults["arena_bounds_m"]
        self.bounds = ArenaBounds(
            x_min=float(bounds["x_min"]),
            x_max=float(bounds["x_max"]),
            y_min=float(bounds["y_min"]),
            y_max=float(bounds["y_max"]),
        )

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        rng = random.Random(seed + 991)

        max_step_count = max_steps or self.default_max_steps
        scenario = self._scenario(seed)
        robot = self.challenge.robot
        world = MujocoWorld.create(
            target=scenario.target,
            obstacles=scenario.obstacles,
            bounds=scenario.bounds.__dict__,
            robot=robot,
        )

        pose = scenario.start
        previous_linear_velocity = 0.0
        poses = [PoseSample(t=0.0, x=pose.x, y=pose.y, yaw=pose.yaw)]
        controls: list[ControlSample] = []
        collisions: list[CollisionEvent] = []
        decisions: list[CollisionDecisionSample] = []
        imu_samples: list[ImuTraceSample] = []
        render_frames: list[RenderFrame] = []
        render_error: str | None = None
        path_length = 0.0
        energy_used = 0.0
        smoothness_cost = 0.0
        last_control: tuple[float, float] | None = None
        first_collision_step: int | None = None
        final_step = 0

        world.reset_robot_pose(pose)
        policy_robot = CollisionProbe(dt=self.dt, max_steps=max_step_count, seed=seed)

        initial_sample = self._imu_sample(
            t=0.0,
            frame_index=0,
            linear_velocity=0.0,
            previous_linear_velocity=0.0,
            angular_velocity=0.0,
            collision_age_sec=None,
            rng=rng,
        )
        imu_samples.append(initial_sample)
        policy_robot._update(time=0.0, imu_sample=self._api_imu_sample(initial_sample))

        try:
            render_stride = max(1, round(self.render_interval_sec / self.dt))
            render_error = self._append_render_frames(
                world=world,
                frames=render_frames,
                t=0.0,
                frame_index=0,
                render_error=render_error,
            )

            for step in range(max_step_count):
                control_time = step * self.dt
                left, right = self._hidden_control(step=step, collision_seen=first_collision_step is not None)
                controls.append(
                    ControlSample(
                        t=round(control_time, 6),
                        left_wheel_velocity=left,
                        right_wheel_velocity=right,
                    )
                )

                if last_control is not None:
                    smoothness_cost += ((left - last_control[0]) ** 2 + (right - last_control[1]) ** 2) * self.dt
                last_control = (left, right)
                energy_used += (left**2 + right**2) * self.dt

                previous_pose = pose
                pose = (
                    world.step_diff_drive(left=left, right=right, wheel_base=robot.wheel_base_m, dt=self.dt)
                    if world.available
                    else self._integrate(pose, left=left, right=right)
                )
                final_step = step + 1
                t = final_step * self.dt
                path_length += math.hypot(pose.x - previous_pose.x, pose.y - previous_pose.y)
                poses.append(PoseSample(t=round(t, 6), x=pose.x, y=pose.y, yaw=pose.yaw))

                collision = None
                if first_collision_step is None:
                    collision = (
                        self._mujoco_collision(t=t, world=world)
                        if world.available
                        else self._first_collision(t=t, pose=pose, scenario=scenario)
                    )
                    analytic_collision = self._first_collision(t=t, pose=pose, scenario=scenario)
                    if collision is None and analytic_collision is not None:
                        collision = analytic_collision
                    if collision is not None:
                        collisions.append(collision)
                        first_collision_step = final_step

                collision_age_sec = (
                    None
                    if first_collision_step is None
                    else max(0.0, t - first_collision_step * self.dt)
                )
                linear_velocity = math.hypot(pose.x - previous_pose.x, pose.y - previous_pose.y) / self.dt
                angular_velocity = _wrap_angle(pose.yaw - previous_pose.yaw) / self.dt
                imu_sample = self._imu_sample(
                    t=t,
                    frame_index=final_step,
                    linear_velocity=linear_velocity,
                    previous_linear_velocity=previous_linear_velocity,
                    angular_velocity=angular_velocity,
                    collision_age_sec=collision_age_sec,
                    rng=rng,
                )
                previous_linear_velocity = linear_velocity
                imu_samples.append(imu_sample)

                policy_robot._update(time=round(t, 6), imu_sample=self._api_imu_sample(imu_sample))
                decision = self._call_policy_step(policy, policy_robot)
                decisions.append(
                    CollisionDecisionSample(
                        t=round(t, 6),
                        frame_index=final_step,
                        contact=decision.contact,
                        severity=decision.severity,  # type: ignore[arg-type]
                    )
                )

                if final_step % render_stride == 0 or collision is not None:
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t,
                        frame_index=final_step,
                        render_error=render_error,
                    )

                if first_collision_step is not None and final_step - first_collision_step >= self.post_collision_steps:
                    break
        finally:
            world.close()

        elapsed = final_step * self.dt
        stats = self._detection_stats(collisions=collisions, decisions=decisions)
        final_distance = math.hypot(scenario.target.x - pose.x, scenario.target.y - pose.y)
        metrics = self._score(
            stats=stats,
            elapsed=elapsed,
            final_distance=final_distance,
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
            imu_samples=imu_samples,
            collision_decisions=decisions,
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "runner": "imu_collision",
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
                "imu_sample_count": len(imu_samples),
                "collision_decision_count": len(decisions),
                "actual_collision_time_sec": stats.actual_collision_time_sec,
                "first_detection_time_sec": stats.first_detection_time_sec,
                "detection_latency_sec": stats.detection_latency_sec,
                "false_positive_count": stats.false_positive_count,
                "detection_success": stats.detected,
                "latency_limit_sec": self.latency_limit_sec,
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

    def _scenario(self, seed: int) -> ImuCollisionScenario:
        rng = random.Random(seed)
        start_y = rng.uniform(-0.08, 0.08)
        start = Pose2D(
            x=-1.75 + rng.uniform(-0.04, 0.04),
            y=start_y,
            yaw=rng.uniform(-0.025, 0.025),
        )
        target = Target(x=1.45, y=start_y, radius=0.18)
        obstacles = [
            Obstacle(
                id="collision_post",
                x=-0.55 + rng.uniform(-0.03, 0.03),
                y=start_y + rng.uniform(-0.025, 0.025),
                radius=0.22,
            )
        ]
        return ImuCollisionScenario(start=start, target=target, obstacles=obstacles, bounds=self.bounds)

    def _hidden_control(self, *, step: int, collision_seen: bool) -> tuple[float, float]:
        if collision_seen:
            return 0.0, 0.0
        base_velocity = 0.58
        dither = 0.015 * math.sin(step * 0.18)
        return base_velocity - dither, base_velocity + dither

    def _call_policy_step(self, policy: RobotPolicyProtocol, robot: CollisionProbe):
        try:
            robot._clear_decision()
            step_result = policy.step(robot)  # type: ignore[arg-type]
            if step_result is not None:
                raise SimulationError(
                    "RobotPolicy.step(robot) should call robot.submit_collision_decision(contact, severity), not return an action."
                )
            return robot._consume_collision_decision()
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

    def _first_collision(self, *, t: float, pose: Pose2D, scenario: ImuCollisionScenario) -> CollisionEvent | None:
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

    def _imu_sample(
        self,
        *,
        t: float,
        frame_index: int,
        linear_velocity: float,
        previous_linear_velocity: float,
        angular_velocity: float,
        collision_age_sec: float | None,
        rng: random.Random,
    ) -> ImuTraceSample:
        normal_accel = _clamp((linear_velocity - previous_linear_velocity) / self.dt, -2.0, 2.0)
        vibration = math.sin(2.0 * math.pi * 11.0 * t)
        terrain_bump = 3.6 * math.exp(-((t - 0.75) / 0.06) ** 2)

        ax = normal_accel + 0.45 * vibration + 0.3 * terrain_bump + rng.gauss(0.0, 0.18)
        ay = 0.16 * math.sin(2.0 * math.pi * 7.0 * t + 0.4) + rng.gauss(0.0, 0.08)
        az = 9.81 + 0.2 * math.sin(2.0 * math.pi * 13.0 * t) + terrain_bump + rng.gauss(0.0, 0.12)
        gx = rng.gauss(0.0, 0.025)
        gy = rng.gauss(0.0, 0.025)
        gz = angular_velocity + rng.gauss(0.0, 0.035)

        if collision_age_sec is not None and collision_age_sec <= 0.16:
            envelope = math.exp(-collision_age_sec / 0.045)
            ax += -18.0 * envelope
            az += 8.5 * envelope
            gz += 3.4 * envelope

        return ImuTraceSample(
            t=round(t, 6),
            frame_index=frame_index,
            ax=round(ax, 6),
            ay=round(ay, 6),
            az=round(az, 6),
            gx=round(gx, 6),
            gy=round(gy, 6),
            gz=round(gz, 6),
        )

    def _api_imu_sample(self, sample: ImuTraceSample) -> ImuSample:
        return ImuSample(
            t=sample.t,
            ax=sample.ax,
            ay=sample.ay,
            az=sample.az,
            gx=sample.gx,
            gy=sample.gy,
            gz=sample.gz,
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

    def _detection_stats(
        self,
        *,
        collisions: list[CollisionEvent],
        decisions: list[CollisionDecisionSample],
    ) -> DetectionStats:
        actual_time = collisions[0].t if collisions else None
        if actual_time is None:
            false_positives = sum(1 for decision in decisions if decision.contact)
            return DetectionStats(
                actual_collision_time_sec=None,
                detected=False,
                detection_latency_sec=None,
                false_positive_count=false_positives,
                first_detection_time_sec=None,
            )

        false_positives = sum(1 for decision in decisions if decision.contact and decision.t < actual_time - self.dt / 2)
        detections = [decision for decision in decisions if decision.contact and decision.t >= actual_time - self.dt / 2]
        first_detection = detections[0] if detections else None
        latency = None if first_detection is None else max(0.0, first_detection.t - actual_time)
        detected = latency is not None and latency <= self.latency_limit_sec
        return DetectionStats(
            actual_collision_time_sec=actual_time,
            detected=detected,
            detection_latency_sec=None if latency is None else round(latency, 6),
            false_positive_count=false_positives,
            first_detection_time_sec=None if first_detection is None else first_detection.t,
        )

    def _score(
        self,
        *,
        stats: DetectionStats,
        elapsed: float,
        final_distance: float,
        collisions: int,
        path_length: float,
        energy_used: float,
        smoothness_cost: float,
    ) -> ScoreMetrics:
        latency = stats.detection_latency_sec
        detection_points = 60.0 if stats.detected else 0.0
        latency_points = (
            20.0 * max(0.0, 1.0 - latency / self.latency_limit_sec)
            if latency is not None and stats.detected
            else 0.0
        )
        precision_points = max(0.0, 20.0 - 10.0 * stats.false_positive_count)
        total = min(100.0, detection_points + latency_points + precision_points)
        success = stats.detected and stats.false_positive_count == 0

        return ScoreMetrics(
            score=round(total, 3),
            success=success,
            status="success" if success else "timeout",
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(final_distance, 6),
            collision_count=collisions,
            path_length_m=round(path_length, 6),
            energy_used=round(energy_used, 6),
            smoothness_cost=round(smoothness_cost, 6),
        )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
