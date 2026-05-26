from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from roborank_envs.catalog import TRAPEZOIDAL_MOTION_PROFILE
from roborank_envs.models import (
    ChallengeSpec,
    MotionProfileControlSample,
    MotionProfileStateSample,
    ReplayTrace,
    RenderFrame,
    RunResult,
    ScoreMetrics,
)
from roborank_envs.policy_api import MotionState1D as ApiMotionState1D
from roborank_envs.policy_api import ProfiledCart1D as ProfiledCartApi
from roborank_envs.policy_api import RobotPolicyProtocol
from roborank_envs.simulation.profiled_cart_world import ProfiledCartWorld
from roborank_envs.simulation.rendering import configured_render_interval_sec
from roborank_envs.simulation.rerun_export import write_rerun_recording


LIMIT_EPS = 1e-6


@dataclass(frozen=True)
class MotionProfileScenario:
    initial_position_m: float
    initial_velocity_mps: float
    target_position_m: float
    target_velocity_mps: float = 0.0


@dataclass(frozen=True)
class MotionProfilePlantState:
    position_m: float
    velocity_mps: float
    acceleration_mps2: float


class SimulationError(RuntimeError):
    pass


class MotionProfileRunner:
    def __init__(self, challenge: ChallengeSpec = TRAPEZOIDAL_MOTION_PROFILE) -> None:
        self.challenge = challenge
        self.dt = float(challenge.defaults["dt_sec"])
        self.default_max_steps = int(challenge.defaults["max_steps"])
        self.render_interval_sec = configured_render_interval_sec(
            float(challenge.defaults.get("render_interval_sec", 0.1))
        )
        if challenge.robot.type != "profiled_cart_1d":
            raise ValueError("MotionProfileRunner requires a profiled_cart_1d challenge.")

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        cart = self.challenge.robot
        max_step_count = max_steps or self.default_max_steps
        scenario = self._scenario(seed)
        world = ProfiledCartWorld.create(cart=cart)
        state = MotionProfilePlantState(
            position_m=scenario.initial_position_m,
            velocity_mps=scenario.initial_velocity_mps,
            acceleration_mps2=0.0,
        )
        policy_cart = ProfiledCartApi(
            max_velocity_mps=cart.max_velocity_mps,
            max_acceleration_mps2=cart.max_acceleration_mps2,
            target_position_m=scenario.target_position_m,
            target_velocity_mps=scenario.target_velocity_mps,
            dt=self.dt,
            max_steps=max_step_count,
            seed=seed,
        )

        states: list[MotionProfileStateSample] = []
        controls: list[MotionProfileControlSample] = []
        render_frames: list[RenderFrame] = []
        render_error: str | None = None
        rerun_error: str | None = None
        energy_used = 0.0
        smoothness_cost = 0.0
        path_length = 0.0
        acceleration_limit_violations = 0
        velocity_limit_violations = 0
        last_command: float | None = None
        status = "timeout"
        final_step = 0

        world.update(
            cart_position_m=state.position_m,
            target_position_m=scenario.target_position_m,
        )
        states.append(self._state_sample(t=0.0, frame_index=0, state=state, scenario=scenario, command=None))

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
                policy_cart._update(
                    time=round(t, 6),
                    state=ApiMotionState1D(
                        t=round(t, 6),
                        position_m=state.position_m,
                        velocity_mps=state.velocity_mps,
                        acceleration_mps2=state.acceleration_mps2,
                    ),
                )
                command = self._call_policy_step(policy, policy_cart)
                acceleration_violation = abs(command) > cart.max_acceleration_mps2 + LIMIT_EPS
                if acceleration_violation:
                    acceleration_limit_violations += 1
                applied_acceleration = _clamp(
                    command,
                    -cart.max_acceleration_mps2,
                    cart.max_acceleration_mps2,
                )
                controls.append(
                    MotionProfileControlSample(
                        t=round(t, 6),
                        acceleration_command_mps2=round(command, 6),
                        applied_acceleration_mps2=round(applied_acceleration, 6),
                        acceleration_limit_violation=acceleration_violation,
                    )
                )

                if last_command is not None:
                    smoothness_cost += ((command - last_command) ** 2) * self.dt
                last_command = command
                energy_used += (applied_acceleration**2) * self.dt

                previous_position = state.position_m
                state, velocity_violation = self._integrate(state=state, acceleration=applied_acceleration)
                if velocity_violation:
                    velocity_limit_violations += 1
                final_step = step + 1
                t_next = final_step * self.dt
                path_length += abs(state.position_m - previous_position)

                world.update(
                    cart_position_m=state.position_m,
                    target_position_m=scenario.target_position_m,
                )
                states.append(
                    self._state_sample(
                        t=t_next,
                        frame_index=final_step,
                        state=state,
                        scenario=scenario,
                        command=command,
                        acceleration_limit_violation=acceleration_violation,
                        velocity_limit_violation=velocity_violation,
                    )
                )

                if final_step % render_stride == 0:
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t_next,
                        frame_index=final_step,
                        render_error=render_error,
                    )

                if self._inside_goal_tolerance(state=state, scenario=scenario):
                    status = "success"
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t_next,
                        frame_index=final_step,
                        render_error=render_error,
                    )
                    break
        finally:
            world.close()

        elapsed = final_step * self.dt
        metrics = self._score(
            status=status,
            elapsed=elapsed,
            states=states,
            controls=controls,
            scenario=scenario,
            path_length=path_length,
            energy_used=energy_used,
            smoothness_cost=smoothness_cost,
            acceleration_limit_violations=acceleration_limit_violations,
            velocity_limit_violations=velocity_limit_violations,
        )
        replay = ReplayTrace(
            motion_profile_controls=controls,
            motion_profile_states=states,
            render_frames=render_frames,
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "runner": "trapezoidal_motion_profile",
                "challenge_mode": "trapezoidal_motion_profile",
                "mujoco_backend": world.backend_name,
                "simulation_mode": "analytic_profiled_cart_with_mujoco_render"
                if world.available
                else "analytic_profiled_cart_fallback",
                "mujoco_timestep_sec": world.timestep if world.available else None,
                "mujoco_render_frames": len(render_frames),
                "mujoco_render_cameras": list(world.render_cameras),
                "mujoco_render_interval_sec": self.render_interval_sec if world.available else None,
                "mujoco_render_resolution": {
                    "width": world.render_width,
                    "height": world.render_height,
                },
                "mujoco_render_error": render_error,
                "robot_model": cart.model,
                "motion_limits": {
                    "max_velocity_mps": cart.max_velocity_mps,
                    "max_acceleration_mps2": cart.max_acceleration_mps2,
                },
                "initial_state": states[0].model_dump(),
                "target_position_m": scenario.target_position_m,
                "target_velocity_mps": scenario.target_velocity_mps,
                "optimal_time_sec": metrics.optimal_time_sec,
                "track_half_width_m": cart.track_half_width_m,
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

    def _scenario(self, seed: int) -> MotionProfileScenario:
        rng = random.Random(seed)
        distance_min, distance_max = self.challenge.defaults["target_distance_range_m"]
        distance = rng.uniform(float(distance_min), float(distance_max))
        if self.challenge.defaults.get("allow_negative_targets", False) and rng.random() < 0.5:
            distance = -distance
        return MotionProfileScenario(
            initial_position_m=0.0,
            initial_velocity_mps=0.0,
            target_position_m=round(distance, 3),
            target_velocity_mps=0.0,
        )

    def _call_policy_step(self, policy: RobotPolicyProtocol, cart: ProfiledCartApi) -> float:
        try:
            cart._clear_command()
            step_result = policy.step(cart)
            if step_result is not None:
                raise SimulationError(
                    "RobotPolicy.step(cart) should call cart.set_acceleration(acceleration_mps2), not return an action."
                )
            return cart._consume_acceleration_command()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _integrate(
        self,
        *,
        state: MotionProfilePlantState,
        acceleration: float,
    ) -> tuple[MotionProfilePlantState, bool]:
        cart = self.challenge.robot
        unclamped_velocity = state.velocity_mps + acceleration * self.dt
        velocity_violation = abs(unclamped_velocity) > cart.max_velocity_mps + LIMIT_EPS
        velocity = _clamp(unclamped_velocity, -cart.max_velocity_mps, cart.max_velocity_mps)
        position = state.position_m + 0.5 * (state.velocity_mps + velocity) * self.dt
        position = _clamp(position, -cart.track_half_width_m, cart.track_half_width_m)
        return (
            MotionProfilePlantState(
                position_m=position,
                velocity_mps=velocity,
                acceleration_mps2=acceleration,
            ),
            velocity_violation,
        )

    def _state_sample(
        self,
        *,
        t: float,
        frame_index: int,
        state: MotionProfilePlantState,
        scenario: MotionProfileScenario,
        command: float | None,
        acceleration_limit_violation: bool = False,
        velocity_limit_violation: bool = False,
    ) -> MotionProfileStateSample:
        position_error = scenario.target_position_m - state.position_m
        velocity_error = scenario.target_velocity_mps - state.velocity_mps
        return MotionProfileStateSample(
            t=round(t, 6),
            frame_index=frame_index,
            position_m=round(state.position_m, 6),
            velocity_mps=round(state.velocity_mps, 6),
            acceleration_mps2=round(state.acceleration_mps2, 6),
            target_position_m=round(scenario.target_position_m, 6),
            target_velocity_mps=round(scenario.target_velocity_mps, 6),
            position_error_m=round(position_error, 6),
            velocity_error_mps=round(velocity_error, 6),
            acceleration_command_mps2=round(command, 6) if command is not None else None,
            acceleration_limit_violation=acceleration_limit_violation,
            velocity_limit_violation=velocity_limit_violation,
        )

    def _inside_goal_tolerance(
        self,
        *,
        state: MotionProfilePlantState,
        scenario: MotionProfileScenario,
    ) -> bool:
        position_tolerance = float(self.challenge.success_conditions["position_tolerance_m"])
        velocity_tolerance = float(self.challenge.success_conditions["velocity_tolerance_mps"])
        return (
            abs(scenario.target_position_m - state.position_m) <= position_tolerance
            and abs(scenario.target_velocity_mps - state.velocity_mps) <= velocity_tolerance
        )

    def _append_render_frames(
        self,
        *,
        world: ProfiledCartWorld,
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

    def _score(
        self,
        *,
        status: str,
        elapsed: float,
        states: list[MotionProfileStateSample],
        controls: list[MotionProfileControlSample],
        scenario: MotionProfileScenario,
        path_length: float,
        energy_used: float,
        smoothness_cost: float,
        acceleration_limit_violations: int,
        velocity_limit_violations: int,
    ) -> ScoreMetrics:
        final = states[-1]
        position_tolerance = float(self.challenge.success_conditions["position_tolerance_m"])
        velocity_tolerance = float(self.challenge.success_conditions["velocity_tolerance_mps"])
        max_time_ratio = float(self.challenge.success_conditions["max_time_ratio_vs_optimal"])
        max_time = self.default_max_steps * self.dt
        optimal_time = _optimal_time(
            distance_m=abs(scenario.target_position_m - scenario.initial_position_m),
            max_velocity_mps=self.challenge.robot.max_velocity_mps,
            max_acceleration_mps2=self.challenge.robot.max_acceleration_mps2,
        )
        final_position_error = abs(final.position_error_m)
        final_velocity_error = abs(final.velocity_error_mps)
        finish_time = elapsed if status == "success" else None
        time_limit = max_time_ratio * optimal_time + self.dt

        success = (
            status == "success"
            and final_position_error <= position_tolerance
            and final_velocity_error <= velocity_tolerance
            and acceleration_limit_violations == 0
            and velocity_limit_violations == 0
            and elapsed <= time_limit
        )

        if not success:
            if acceleration_limit_violations or velocity_limit_violations:
                status = "limit_violation"
            elif status == "success" or final_position_error > position_tolerance or final_velocity_error > velocity_tolerance:
                status = "accuracy_error"
            else:
                status = "timeout"

        accuracy_points = 35.0 * _clamp01(1.0 - final_position_error / max(position_tolerance * 6.0, 1e-9))
        velocity_points = 20.0 * _clamp01(1.0 - final_velocity_error / max(velocity_tolerance * 6.0, 1e-9))
        limit_points = 20.0
        limit_points -= min(20.0, 8.0 * acceleration_limit_violations + 5.0 * velocity_limit_violations)
        if status == "timeout":
            time_ratio = max_time / max(optimal_time, 1e-9)
        else:
            time_ratio = elapsed / max(optimal_time, 1e-9)
        time_points = 20.0 * _clamp01(1.0 - max(0.0, time_ratio - 1.0) / max(max_time_ratio - 1.0, 1e-9))
        smoothness_points = max(0.0, 5.0 - 0.16 * smoothness_cost)
        total = min(100.0, accuracy_points + velocity_points + limit_points + time_points + smoothness_points)

        max_abs_velocity = max((abs(sample.velocity_mps) for sample in states), default=0.0)
        max_abs_command = max((abs(sample.acceleration_command_mps2) for sample in controls), default=0.0)

        return ScoreMetrics(
            metric_kind="motion_profile",
            score=round(total, 3),
            success=success,
            status=("success" if success else status),  # type: ignore[arg-type]
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(final_position_error, 6),
            collision_count=0,
            path_length_m=round(path_length, 6),
            energy_used=round(energy_used, 6),
            smoothness_cost=round(smoothness_cost, 6),
            final_position_error_m=round(final_position_error, 6),
            mean_position_error_m=round(
                sum(abs(sample.position_error_m) for sample in states) / max(1, len(states)),
                6,
            ),
            max_position_error_m=round(max(abs(sample.position_error_m) for sample in states), 6),
            final_velocity_error_mps=round(final_velocity_error, 6),
            max_abs_velocity_mps=round(max_abs_velocity, 6),
            max_abs_acceleration_command_mps2=round(max_abs_command, 6),
            acceleration_limit_violation_count=acceleration_limit_violations,
            velocity_limit_violation_count=velocity_limit_violations,
            optimal_time_sec=round(optimal_time, 6),
            finish_time_sec=round(finish_time, 6) if finish_time is not None else None,
        )


def _optimal_time(*, distance_m: float, max_velocity_mps: float, max_acceleration_mps2: float) -> float:
    distance = max(0.0, distance_m)
    vmax = max(max_velocity_mps, 1e-9)
    accel = max(max_acceleration_mps2, 1e-9)
    distance_with_cruise = (vmax * vmax) / accel
    if distance >= distance_with_cruise:
        return 2.0 * vmax / accel + (distance - distance_with_cruise) / vmax
    return 2.0 * math.sqrt(distance / accel)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
