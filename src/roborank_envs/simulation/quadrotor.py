from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from roborank_envs.catalog import QUADROTOR_GATE_SEQUENCE
from roborank_envs.models import (
    ChallengeSpec,
    CollisionEvent,
    ControlSample,
    PoseSample,
    ReplayTrace,
    RenderFrame,
    RunResult,
    ScoreMetrics,
    Target,
)
from roborank_envs.policy_api import Gate3d as ApiGate3d
from roborank_envs.policy_api import Pose3d as ApiPose3d
from roborank_envs.policy_api import Quadrotor as QuadrotorApi
from roborank_envs.policy_api import RobotPolicyProtocol
from roborank_envs.simulation.quadrotor_world import GateVisual, QuadrotorMujocoWorld
from roborank_envs.simulation.rendering import configured_render_interval_sec
from roborank_envs.simulation.rerun_export import write_rerun_recording

GRAVITY_MPS2 = 9.81
DEFAULT_REPLAY_RENDER_INTERVAL_SEC = 0.1


@dataclass(frozen=True)
class FlightVolume:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class Gate:
    id: str
    x: float
    y: float
    z: float
    yaw: float
    width_m: float
    height_m: float


@dataclass(frozen=True)
class QuadrotorState:
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
class QuadrotorScenario:
    start: QuadrotorState
    gates: list[Gate]
    bounds: FlightVolume


class SimulationError(RuntimeError):
    pass


class QuadrotorGateRunner:
    def __init__(self, challenge: ChallengeSpec = QUADROTOR_GATE_SEQUENCE) -> None:
        self.challenge = challenge
        self.dt = float(challenge.defaults["dt_sec"])
        self.default_max_steps = int(challenge.defaults["max_steps"])
        self.render_interval_sec = configured_render_interval_sec(
            float(challenge.defaults.get("render_interval_sec", DEFAULT_REPLAY_RENDER_INTERVAL_SEC))
        )
        bounds = challenge.defaults["flight_volume_m"]
        self.bounds = FlightVolume(
            x_min=float(bounds["x_min"]),
            x_max=float(bounds["x_max"]),
            y_min=float(bounds["y_min"]),
            y_max=float(bounds["y_max"]),
            z_min=float(bounds["z_min"]),
            z_max=float(bounds["z_max"]),
        )
        self.gate_margin = float(challenge.success_conditions["gate_aperture_margin_m"])
        self.max_tilt_rad = math.radians(float(challenge.success_conditions["max_tilt_deg"]))

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        max_step_count = max_steps or self.default_max_steps
        scenario = self._scenario(seed)
        robot = self.challenge.robot
        world = QuadrotorMujocoWorld.create(
            gates=[self._gate_visual(gate) for gate in scenario.gates],
            bounds=scenario.bounds.__dict__,
            robot=robot,
        )

        state = scenario.start
        gates_completed = 0
        gate_pass_times: list[float] = []
        poses = [self._pose_sample(t=0.0, state=state)]
        controls: list[ControlSample] = []
        collisions: list[CollisionEvent] = []
        render_frames: list[RenderFrame] = []
        render_error: str | None = None
        path_length = 0.0
        energy_used = 0.0
        smoothness_cost = 0.0
        max_attitude = _tilt_angle(state.roll, state.pitch)
        last_control: tuple[float, float, float, float] | None = None
        status = "timeout"
        final_step = 0

        world.reset_drone_pose(
            x=state.x,
            y=state.y,
            z=state.z,
            roll=state.roll,
            pitch=state.pitch,
            yaw=state.yaw,
        )
        policy_robot = QuadrotorApi(
            hover_power=float(robot.hover_power),
            max_power=float(robot.max_power),
            max_body_rate_radps=float(robot.max_body_rate_radps),
            max_tilt_rad=float(robot.max_tilt_rad),
            dt=self.dt,
            max_steps=max_step_count,
            seed=seed,
        )
        self._update_policy_robot(
            robot=policy_robot,
            t=0.0,
            state=state,
            scenario=scenario,
            gates_completed=gates_completed,
            collision_count=0,
        )

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
                t = step * self.dt
                self._update_policy_robot(
                    robot=policy_robot,
                    t=t,
                    state=state,
                    scenario=scenario,
                    gates_completed=gates_completed,
                    collision_count=len(collisions),
                )
                command = self._call_policy_step(policy, policy_robot)
                roll_rate, pitch_rate, yaw_rate, power = command
                controls.append(
                    ControlSample(
                        t=round(t, 6),
                        roll_rate_radps=roll_rate,
                        pitch_rate_radps=pitch_rate,
                        yaw_rate_radps=yaw_rate,
                        power=power,
                    )
                )

                if last_control is not None:
                    smoothness_cost += sum((command[index] - last_control[index]) ** 2 for index in range(4)) * self.dt
                last_control = command
                energy_used += (power**2 + 0.05 * (roll_rate**2 + pitch_rate**2 + yaw_rate**2)) * self.dt

                previous_state = state
                state = self._integrate(
                    state,
                    roll_rate=roll_rate,
                    pitch_rate=pitch_rate,
                    yaw_rate=yaw_rate,
                    power=power,
                )
                final_step = step + 1
                t_next = final_step * self.dt
                max_attitude = max(max_attitude, _tilt_angle(state.roll, state.pitch))
                path_length += math.sqrt(
                    (state.x - previous_state.x) ** 2
                    + (state.y - previous_state.y) ** 2
                    + (state.z - previous_state.z) ** 2
                )
                poses.append(self._pose_sample(t=t_next, state=state))
                world.set_drone_pose(
                    x=state.x,
                    y=state.y,
                    z=state.z,
                    roll=state.roll,
                    pitch=state.pitch,
                    yaw=state.yaw,
                )

                collision = self._first_safety_violation(t=t_next, state=state, scenario=scenario)
                if collision is None and gates_completed < len(scenario.gates):
                    gate_event = self._gate_crossing(
                        t=t_next,
                        previous_state=previous_state,
                        state=state,
                        gate=scenario.gates[gates_completed],
                    )
                    if gate_event == "passed":
                        gates_completed += 1
                        gate_pass_times.append(round(t_next, 6))
                    elif isinstance(gate_event, CollisionEvent):
                        collision = gate_event

                should_render = final_step % render_stride == 0 or collision is not None or gates_completed == len(scenario.gates)
                if should_render:
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t_next,
                        frame_index=final_step,
                        render_error=render_error,
                    )

                if collision is not None:
                    collisions.append(collision)
                    status = "collision"
                    break

                if gates_completed == len(scenario.gates):
                    status = "success"
                    break
        finally:
            world.close()

        elapsed = final_step * self.dt
        final_distance = self._distance_to_next_gate(state=state, scenario=scenario, gates_completed=gates_completed)
        metrics = self._score(
            status=status,
            elapsed=elapsed,
            final_distance=final_distance,
            collisions=len(collisions),
            path_length=path_length,
            energy_used=energy_used,
            smoothness_cost=smoothness_cost,
            gates_completed=gates_completed,
            gate_count=len(scenario.gates),
            max_attitude=max_attitude,
        )
        final_gate = scenario.gates[-1]
        replay = ReplayTrace(
            poses=poses,
            target=Target(
                x=final_gate.x,
                y=final_gate.y,
                radius=min(final_gate.width_m, final_gate.height_m) / 2.0,
            ),
            collisions=collisions,
            controls=controls,
            render_frames=render_frames,
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "runner": "quadrotor_gate_sequence",
                "mujoco_backend": world.backend_name,
                "simulation_mode": "analytic_quadrotor_with_mujoco_render" if world.available else "analytic_quadrotor_no_mujoco",
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
                    "arm_length": robot.arm_length_m,
                    "radius": robot.radius_m,
                    "height": robot.height_m,
                    "mass_kg": robot.mass_kg,
                },
                "reference_frames": {
                    "world": "ENU: +x course forward, +y left, +z up.",
                    "body": "+x nose forward, +y left, +z up. Body-rate commands are about these body axes.",
                    "gate": "Gate yaw is the world-frame heading of the gate normal; gates are passed from negative normal side to positive normal side.",
                },
                "flight_volume_m": scenario.bounds.__dict__,
                "gates": [gate.__dict__ for gate in scenario.gates],
                "gates_completed": gates_completed,
                "gate_count": len(scenario.gates),
                "gate_pass_times_sec": gate_pass_times,
                "max_attitude_deg": round(math.degrees(max_attitude), 6),
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

    def _scenario(self, seed: int) -> QuadrotorScenario:
        rng = random.Random(seed)
        start = QuadrotorState(
            x=0.0,
            y=rng.uniform(-0.04, 0.04),
            z=1.02 + rng.uniform(-0.03, 0.03),
            roll=0.0,
            pitch=0.0,
            yaw=rng.uniform(-0.04, 0.04),
            vx=0.0,
            vy=0.0,
            vz=0.0,
        )
        gates = [
            Gate(id="gate_1", x=1.45 + rng.uniform(-0.04, 0.04), y=rng.uniform(-0.08, 0.08), z=1.02, yaw=0.0, width_m=0.95, height_m=0.72),
            Gate(id="gate_2", x=2.65 + rng.uniform(-0.05, 0.05), y=0.28 + rng.uniform(-0.10, 0.10), z=1.12 + rng.uniform(-0.05, 0.05), yaw=rng.uniform(-0.08, 0.08), width_m=0.95, height_m=0.76),
            Gate(id="gate_3", x=3.85 + rng.uniform(-0.06, 0.06), y=-0.24 + rng.uniform(-0.10, 0.10), z=1.04 + rng.uniform(-0.05, 0.05), yaw=rng.uniform(-0.10, 0.10), width_m=0.95, height_m=0.76),
        ]
        return QuadrotorScenario(start=start, gates=gates, bounds=self.bounds)

    def _update_policy_robot(
        self,
        *,
        robot: QuadrotorApi,
        t: float,
        state: QuadrotorState,
        scenario: QuadrotorScenario,
        gates_completed: int,
        collision_count: int,
    ) -> None:
        next_gate = scenario.gates[min(gates_completed, len(scenario.gates) - 1)]
        robot._update(
            time=round(t, 6),
            pose=ApiPose3d(
                x=state.x,
                y=state.y,
                z=state.z,
                roll=state.roll,
                pitch=state.pitch,
                yaw=state.yaw,
                vx=state.vx,
                vy=state.vy,
                vz=state.vz,
            ),
            next_gate=ApiGate3d(
                id=next_gate.id,
                x=next_gate.x,
                y=next_gate.y,
                z=next_gate.z,
                yaw=next_gate.yaw,
                width_m=next_gate.width_m,
                height_m=next_gate.height_m,
            ),
            gates_completed=gates_completed,
            gate_count=len(scenario.gates),
            collision_count=collision_count,
        )

    def _call_policy_step(self, policy: RobotPolicyProtocol, robot: QuadrotorApi) -> tuple[float, float, float, float]:
        try:
            robot._clear_command()
            step_result = policy.step(robot)
            if step_result is not None:
                raise SimulationError(
                    "RobotPolicy.step(robot) should command robot.set_body_rate_and_power(...), not return an action."
                )
            return robot._consume_command()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _integrate(
        self,
        state: QuadrotorState,
        *,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
        power: float,
    ) -> QuadrotorState:
        hover_power = max(1e-6, float(self.challenge.robot.hover_power))
        thrust_accel = (power / hover_power) * GRAVITY_MPS2
        roll = _clamp(state.roll + roll_rate * self.dt, -1.35, 1.35)
        pitch = _clamp(state.pitch + pitch_rate * self.dt, -1.35, 1.35)
        yaw = _wrap_angle(state.yaw + yaw_rate * self.dt)

        body_z = _body_z_axis(roll=roll, pitch=pitch, yaw=yaw)
        velocity = np.array([state.vx, state.vy, state.vz], dtype=float)
        acceleration = thrust_accel * body_z + np.array([0.0, 0.0, -GRAVITY_MPS2]) - 0.24 * velocity
        next_velocity = velocity + acceleration * self.dt
        speed = float(np.linalg.norm(next_velocity))
        if speed > 3.0:
            next_velocity *= 3.0 / speed

        next_position = np.array([state.x, state.y, state.z], dtype=float) + next_velocity * self.dt
        return QuadrotorState(
            x=float(next_position[0]),
            y=float(next_position[1]),
            z=float(next_position[2]),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            vx=float(next_velocity[0]),
            vy=float(next_velocity[1]),
            vz=float(next_velocity[2]),
        )

    def _first_safety_violation(
        self,
        *,
        t: float,
        state: QuadrotorState,
        scenario: QuadrotorScenario,
    ) -> CollisionEvent | None:
        radius = float(self.challenge.robot.radius_m)
        bounds = scenario.bounds
        penetration = max(
            bounds.x_min + radius - state.x,
            state.x - (bounds.x_max - radius),
            bounds.y_min + radius - state.y,
            state.y - (bounds.y_max - radius),
            bounds.z_min - state.z,
            state.z - bounds.z_max,
        )
        if penetration > 0.0:
            return CollisionEvent(
                t=round(t, 6),
                kind="bounds",
                object_id="flight_volume",
                penetration_m=round(penetration, 6),
            )

        tilt = _tilt_angle(state.roll, state.pitch)
        if tilt > self.max_tilt_rad:
            return CollisionEvent(
                t=round(t, 6),
                kind="bounds",
                object_id="attitude_limit",
                penetration_m=round(tilt - self.max_tilt_rad, 6),
            )
        return None

    def _gate_crossing(
        self,
        *,
        t: float,
        previous_state: QuadrotorState,
        state: QuadrotorState,
        gate: Gate,
    ) -> str | CollisionEvent | None:
        previous_signed = self._signed_gate_distance(previous_state, gate)
        current_signed = self._signed_gate_distance(state, gate)
        if previous_signed > 0.0 or current_signed < 0.0 or abs(current_signed - previous_signed) < 1e-9:
            return None

        alpha = _clamp(-previous_signed / (current_signed - previous_signed), 0.0, 1.0)
        cross_x = previous_state.x + alpha * (state.x - previous_state.x)
        cross_y = previous_state.y + alpha * (state.y - previous_state.y)
        cross_z = previous_state.z + alpha * (state.z - previous_state.z)
        lateral_axis = (-math.sin(gate.yaw), math.cos(gate.yaw), 0.0)
        lateral = (cross_x - gate.x) * lateral_axis[0] + (cross_y - gate.y) * lateral_axis[1]
        vertical = cross_z - gate.z
        radius = float(self.challenge.robot.radius_m)
        lateral_margin = gate.width_m / 2.0 - radius - abs(lateral)
        vertical_margin = gate.height_m / 2.0 - radius - abs(vertical)
        if lateral_margin >= 0.0 and vertical_margin >= 0.0:
            return "passed"

        return CollisionEvent(
            t=round(t, 6),
            kind="obstacle",
            object_id=f"{gate.id}_frame",
            penetration_m=round(abs(min(lateral_margin, vertical_margin)), 6),
        )

    def _signed_gate_distance(self, state: QuadrotorState, gate: Gate) -> float:
        normal = (math.cos(gate.yaw), math.sin(gate.yaw), 0.0)
        return (state.x - gate.x) * normal[0] + (state.y - gate.y) * normal[1]

    def _distance_to_next_gate(
        self,
        *,
        state: QuadrotorState,
        scenario: QuadrotorScenario,
        gates_completed: int,
    ) -> float:
        if gates_completed >= len(scenario.gates):
            return 0.0
        gate = scenario.gates[gates_completed]
        return math.sqrt((gate.x - state.x) ** 2 + (gate.y - state.y) ** 2 + (gate.z - state.z) ** 2)

    def _append_render_frames(
        self,
        *,
        world: QuadrotorMujocoWorld,
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

    def _pose_sample(self, *, t: float, state: QuadrotorState) -> PoseSample:
        return PoseSample(
            t=round(t, 6),
            x=state.x,
            y=state.y,
            z=state.z,
            yaw=state.yaw,
            roll=state.roll,
            pitch=state.pitch,
            vx=state.vx,
            vy=state.vy,
            vz=state.vz,
            speed=math.sqrt(state.vx**2 + state.vy**2 + state.vz**2),
        )

    def _gate_visual(self, gate: Gate) -> GateVisual:
        return GateVisual(
            id=gate.id,
            x=gate.x,
            y=gate.y,
            z=gate.z,
            yaw=gate.yaw,
            width_m=gate.width_m,
            height_m=gate.height_m,
        )

    def _score(
        self,
        *,
        status: str,
        elapsed: float,
        final_distance: float,
        collisions: int,
        path_length: float,
        energy_used: float,
        smoothness_cost: float,
        gates_completed: int,
        gate_count: int,
        max_attitude: float,
    ) -> ScoreMetrics:
        max_time = self.default_max_steps * self.dt
        progress = gates_completed / max(1, gate_count)
        success = status == "success" and collisions == 0 and gates_completed == gate_count

        success_points = 55.0 if success else 35.0 * progress
        time_points = 15.0 * max(0.0, 1.0 - elapsed / max_time) if success else 0.0
        gate_points = 15.0 * progress
        attitude_margin = max(0.0, 1.0 - max(0.0, max_attitude - 0.65 * self.max_tilt_rad) / (0.35 * self.max_tilt_rad))
        safety_points = (10.0 * attitude_margin) if collisions == 0 else 0.0
        smooth_energy_points = max(0.0, 5.0 - 0.18 * smoothness_cost - 0.05 * energy_used)
        total = min(100.0, success_points + time_points + gate_points + safety_points + smooth_energy_points)

        return ScoreMetrics(
            metric_kind="gate_sequence",
            score=round(total, 3),
            success=success,
            status=status,  # type: ignore[arg-type]
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(final_distance, 6),
            collision_count=collisions,
            path_length_m=round(path_length, 6),
            energy_used=round(energy_used, 6),
            smoothness_cost=round(smoothness_cost, 6),
            gates_completed=gates_completed,
            gate_count=gate_count,
            max_attitude_deg=round(math.degrees(max_attitude), 6),
        )


def _body_z_axis(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            cy * sp * cr + sy * sr,
            sy * sp * cr - cy * sr,
            cp * cr,
        ],
        dtype=float,
    )


def _tilt_angle(roll: float, pitch: float) -> float:
    return math.acos(_clamp(math.cos(roll) * math.cos(pitch), -1.0, 1.0))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
