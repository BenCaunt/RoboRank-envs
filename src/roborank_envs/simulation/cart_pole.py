from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from roborank_envs.catalog import CART_POLE
from roborank_envs.models import (
    CartPoleControlSample,
    CartPoleStateSample,
    ChallengeSpec,
    CollisionEvent,
    ReplayTrace,
    RenderFrame,
    RunResult,
    ScoreMetrics,
)
from roborank_envs.policy_api import CartPole as CartPoleApi
from roborank_envs.policy_api import CartPoleState as ApiCartPoleState
from roborank_envs.policy_api import RobotPolicyProtocol
from roborank_envs.simulation.cart_pole_world import CartPoleWorld
from roborank_envs.simulation.rendering import configured_render_interval_sec
from roborank_envs.simulation.rerun_export import write_rerun_recording


FAILURE_ANGLE_RAD = 0.70


@dataclass(frozen=True)
class CartPolePlantState:
    x: float
    x_dot: float
    theta: float
    theta_dot: float


@dataclass(frozen=True)
class CartPoleScenario:
    initial_state: CartPolePlantState
    target_position_m: float = 0.0


class SimulationError(RuntimeError):
    pass


class CartPoleRunner:
    def __init__(self, challenge: ChallengeSpec = CART_POLE) -> None:
        self.challenge = challenge
        self.dt = float(challenge.defaults["dt_sec"])
        self.default_max_steps = int(challenge.defaults["max_steps"])
        self.render_interval_sec = configured_render_interval_sec(
            float(challenge.defaults.get("render_interval_sec", 0.1))
        )
        if challenge.robot.type != "cart_pole":
            raise ValueError("CartPoleRunner requires a cart_pole challenge.")

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        plant = self.challenge.robot
        max_step_count = max_steps or self.default_max_steps
        scenario = self._scenario(seed)
        world = CartPoleWorld.create(plant=plant)
        state = scenario.initial_state
        target = scenario.target_position_m
        policy_cart = CartPoleApi(
            cart_mass_kg=plant.cart_mass_kg,
            pole_mass_kg=plant.pole_mass_kg,
            pole_com_length_m=plant.pole_com_length_m,
            pole_length_m=plant.pole_length_m,
            track_half_width_m=plant.track_half_width_m,
            max_force_n=plant.max_force_n,
            gravity_mps2=plant.gravity_mps2,
            minimum_phase_gain_m=plant.minimum_phase_gain_m,
            target_position_m=target,
            dt=self.dt,
            max_steps=max_step_count,
            seed=seed,
        )

        states: list[CartPoleStateSample] = []
        controls: list[CartPoleControlSample] = []
        collisions: list[CollisionEvent] = []
        render_frames: list[RenderFrame] = []
        render_error: str | None = None
        rerun_error: str | None = None
        energy_used = 0.0
        smoothness_cost = 0.0
        path_length = 0.0
        last_force: float | None = None
        status = "timeout"
        final_step = 0

        world.update(cart_position_m=state.x, pole_angle_rad=state.theta)
        states.append(self._state_sample(t=0.0, frame_index=0, state=state, target=target, force=0.0))

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
                    state=ApiCartPoleState(
                        t=round(t, 6),
                        cart_position_m=state.x,
                        cart_velocity_mps=state.x_dot,
                        pole_angle_rad=state.theta,
                        pole_angular_velocity_radps=state.theta_dot,
                    ),
                )
                commanded_force = self._call_policy_step(policy, policy_cart)
                applied_force = commanded_force + self._disturbance_force(t)
                controls.append(CartPoleControlSample(t=round(t, 6), force_n=round(commanded_force, 6)))

                if last_force is not None:
                    smoothness_cost += ((commanded_force - last_force) ** 2) * self.dt
                last_force = commanded_force
                energy_used += (commanded_force**2) * self.dt

                previous_x = state.x
                state = self._integrate(state=state, force=applied_force)
                final_step = step + 1
                t_next = final_step * self.dt
                path_length += abs(state.x - previous_x)

                world.update(cart_position_m=state.x, pole_angle_rad=state.theta)
                states.append(
                    self._state_sample(
                        t=t_next,
                        frame_index=final_step,
                        state=state,
                        target=target,
                        force=commanded_force,
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

                collision = self._failure_event(t=t_next, state=state)
                if collision is not None:
                    collisions.append(collision)
                    status = "collision"
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t_next,
                        frame_index=final_step,
                        render_error=render_error,
                    )
                    break
            else:
                status = "success"
        finally:
            world.close()

        elapsed = final_step * self.dt
        metrics = self._score(
            status=status,
            elapsed=elapsed,
            states=states,
            collisions=len(collisions),
            path_length=path_length,
            energy_used=energy_used,
            smoothness_cost=smoothness_cost,
        )
        replay = ReplayTrace(
            collisions=collisions,
            cart_pole_controls=controls,
            cart_pole_states=states,
            render_frames=render_frames,
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "runner": "cart_pole",
                "challenge_mode": self.challenge.defaults.get("scenario", "stabilization"),
                "mujoco_backend": world.backend_name,
                "simulation_mode": "analytic_cart_pole_with_mujoco_render" if world.available else "analytic_cart_pole_fallback",
                "mujoco_timestep_sec": world.timestep if world.available else None,
                "mujoco_render_frames": len(render_frames),
                "mujoco_render_cameras": list(world.render_cameras),
                "mujoco_render_interval_sec": self.render_interval_sec if world.available else None,
                "mujoco_render_resolution": {
                    "width": world.render_width,
                    "height": world.render_height,
                },
                "mujoco_render_error": render_error,
                "robot_model": plant.model,
                "cart_pole_parameters": {
                    "cart_mass_kg": plant.cart_mass_kg,
                    "pole_mass_kg": plant.pole_mass_kg,
                    "pole_com_length_m": plant.pole_com_length_m,
                    "pole_length_m": plant.pole_length_m,
                    "track_half_width_m": plant.track_half_width_m,
                    "max_force_n": plant.max_force_n,
                    "gravity_mps2": plant.gravity_mps2,
                    "minimum_phase_gain_m": plant.minimum_phase_gain_m,
                },
                "initial_state": states[0].model_dump(),
                "target_position_m": target,
                "failure_angle_rad": FAILURE_ANGLE_RAD,
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

    def _scenario(self, seed: int) -> CartPoleScenario:
        rng = random.Random(seed)
        if self.challenge.defaults.get("scenario") == "minimum_phase":
            angle_min, angle_max = self.challenge.defaults["initial_pole_angle_abs_range_rad"]
            sign = -1.0 if rng.random() < 0.5 else 1.0
            theta = sign * rng.uniform(float(angle_min), float(angle_max))
        else:
            angle_min, angle_max = self.challenge.defaults["initial_pole_angle_range_rad"]
            theta = rng.uniform(float(angle_min), float(angle_max))

        x_min, x_max = self.challenge.defaults["initial_cart_position_range_m"]
        xd_min, xd_max = self.challenge.defaults["initial_cart_velocity_range_mps"]
        thd_min, thd_max = self.challenge.defaults["initial_pole_angular_velocity_range_radps"]
        return CartPoleScenario(
            initial_state=CartPolePlantState(
                x=rng.uniform(float(x_min), float(x_max)),
                x_dot=rng.uniform(float(xd_min), float(xd_max)),
                theta=theta,
                theta_dot=rng.uniform(float(thd_min), float(thd_max)),
            )
        )

    def _call_policy_step(self, policy: RobotPolicyProtocol, cart_pole: CartPoleApi) -> float:
        try:
            cart_pole._clear_command()
            step_result = policy.step(cart_pole)
            if step_result is not None:
                raise SimulationError("RobotPolicy.step(cart_pole) should call cart_pole.set_force(newtons), not return an action.")
            return cart_pole._consume_force_command()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _integrate(self, *, state: CartPolePlantState, force: float) -> CartPolePlantState:
        plant = self.challenge.robot
        x = state.x
        x_dot = state.x_dot
        theta = state.theta
        theta_dot = state.theta_dot
        clamped_force = _clamp(force, -plant.max_force_n, plant.max_force_n)
        total_mass = plant.cart_mass_kg + plant.pole_mass_kg
        pole_mass_length = plant.pole_mass_kg * plant.pole_com_length_m
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        temp = (clamped_force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
        denominator = plant.pole_com_length_m * (
            4.0 / 3.0 - plant.pole_mass_kg * cos_theta**2 / total_mass
        )
        theta_acc = (plant.gravity_mps2 * sin_theta - cos_theta * temp) / denominator
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

        x_dot += x_acc * self.dt
        x += x_dot * self.dt
        theta_dot += theta_acc * self.dt
        theta = _wrap_angle(theta + theta_dot * self.dt)
        return CartPolePlantState(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot)

    def _state_sample(
        self,
        *,
        t: float,
        frame_index: int,
        state: CartPolePlantState,
        target: float,
        force: float | None,
    ) -> CartPoleStateSample:
        y, y_dot = self._minimum_phase_output(state)
        return CartPoleStateSample(
            t=round(t, 6),
            frame_index=frame_index,
            cart_position_m=round(state.x, 6),
            cart_velocity_mps=round(state.x_dot, 6),
            pole_angle_rad=round(state.theta, 6),
            pole_angular_velocity_radps=round(state.theta_dot, 6),
            target_position_m=round(target, 6),
            minimum_phase_output_m=round(y, 6),
            minimum_phase_output_velocity_mps=round(y_dot, 6),
            force_n=round(force, 6) if force is not None else None,
        )

    def _minimum_phase_output(self, state: CartPolePlantState) -> tuple[float, float]:
        gain = self.challenge.robot.minimum_phase_gain_m
        output = state.x + gain * math.sin(state.theta)
        output_dot = state.x_dot + gain * state.theta_dot * math.cos(state.theta)
        return output, output_dot

    def _disturbance_force(self, t: float) -> float:
        if self.challenge.defaults.get("scenario") != "minimum_phase":
            return 0.0
        start = float(self.challenge.defaults.get("disturbance_start_sec", 0.0))
        duration = float(self.challenge.defaults.get("disturbance_duration_sec", 0.0))
        peak = float(self.challenge.defaults.get("disturbance_peak_force_n", 0.0))
        if duration <= 0.0 or not start <= t <= start + duration:
            return 0.0
        phase = (t - start) / duration
        return peak * math.sin(math.pi * phase)

    def _failure_event(self, *, t: float, state: CartPolePlantState) -> CollisionEvent | None:
        track_limit = float(self.challenge.success_conditions["track_half_width_m"])
        if abs(state.x) > track_limit:
            return CollisionEvent(
                t=round(t, 6),
                kind="bounds",
                object_id="track_limit",
                penetration_m=round(abs(state.x) - track_limit, 6),
            )
        if abs(state.theta) > FAILURE_ANGLE_RAD:
            return CollisionEvent(
                t=round(t, 6),
                kind="obstacle",
                object_id="pole_angle_limit",
                penetration_m=round(abs(state.theta) - FAILURE_ANGLE_RAD, 6),
            )
        return None

    def _append_render_frames(
        self,
        *,
        world: CartPoleWorld,
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
        states: list[CartPoleStateSample],
        collisions: int,
        path_length: float,
        energy_used: float,
        smoothness_cost: float,
    ) -> ScoreMetrics:
        final = states[-1]
        angles = [abs(sample.pole_angle_rad) for sample in states]
        cart_positions = [abs(sample.cart_position_m) for sample in states]
        minimum_phase_outputs = [abs(sample.minimum_phase_output_m or 0.0) for sample in states]
        final_angle = abs(final.pole_angle_rad)
        final_cart = abs(final.cart_position_m)
        final_output = abs(final.minimum_phase_output_m or 0.0)
        max_angle = max(angles) if angles else 0.0
        max_cart = max(cart_positions) if cart_positions else 0.0
        rms_angle = math.sqrt(sum(angle * angle for angle in angles) / max(1, len(angles)))
        rms_output = math.sqrt(sum(output * output for output in minimum_phase_outputs) / max(1, len(minimum_phase_outputs)))
        max_time = self.default_max_steps * self.dt
        survived = status != "collision" and collisions == 0 and elapsed >= max_time - 1e-9

        angle_tolerance = float(self.challenge.success_conditions["final_pole_angle_rad"])
        max_angle_tolerance = float(self.challenge.success_conditions["max_abs_pole_angle_rad"])
        cart_tolerance = float(self.challenge.success_conditions["final_cart_position_m"])
        output_tolerance = self.challenge.success_conditions.get("final_minimum_phase_output_m")
        rms_output_tolerance = self.challenge.success_conditions.get("rms_minimum_phase_output_m")

        success = (
            survived
            and final_angle <= angle_tolerance
            and max_angle <= max_angle_tolerance
            and final_cart <= cart_tolerance
        )
        if output_tolerance is not None:
            success = success and final_output <= float(output_tolerance)
        if rms_output_tolerance is not None:
            success = success and rms_output <= float(rms_output_tolerance)

        if not success and status == "success":
            status = "timeout"

        survival_ratio = _clamp01(elapsed / max(max_time, 1e-9))
        if self.challenge.defaults.get("scenario") == "minimum_phase":
            survival_points = 35.0 * survival_ratio if not survived else 35.0
            output_points = 12.5 * _clamp01(1.0 - final_output / max(float(output_tolerance or 0.12) * 3.0, 1e-9))
            output_points += 12.5 * _clamp01(1.0 - rms_output / max(float(rms_output_tolerance or 0.18) * 2.5, 1e-9))
            pole_points = 10.0 * _clamp01(1.0 - final_angle / max(angle_tolerance * 3.0, 1e-9))
            pole_points += 10.0 * _clamp01(1.0 - rms_angle / max(angle_tolerance * 2.8, 1e-9))
            cart_points = 10.0 * _clamp01(1.0 - final_cart / max(cart_tolerance * 4.0, 1e-9))
            effort_points = max(0.0, 10.0 - 0.018 * energy_used - 0.05 * smoothness_cost)
            total = min(100.0, survival_points + output_points + pole_points + cart_points + effort_points)
        else:
            survival_points = 45.0 * survival_ratio if not survived else 45.0
            pole_points = 12.5 * _clamp01(1.0 - final_angle / max(angle_tolerance * 3.0, 1e-9))
            pole_points += 12.5 * _clamp01(1.0 - rms_angle / max(angle_tolerance * 3.0, 1e-9))
            cart_points = 15.0 * _clamp01(1.0 - final_cart / max(cart_tolerance * 3.0, 1e-9))
            energy_points = max(0.0, 10.0 - 0.02 * energy_used)
            smoothness_points = max(0.0, 5.0 - 0.05 * smoothness_cost)
            total = min(100.0, survival_points + pole_points + cart_points + energy_points + smoothness_points)

        return ScoreMetrics(
            metric_kind="cart_pole",
            score=round(total, 3),
            success=success,
            status=("success" if success else status),  # type: ignore[arg-type]
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(final_cart, 6),
            heading_error_rad=round(final_angle, 6),
            collision_count=collisions,
            path_length_m=round(path_length, 6),
            energy_used=round(energy_used, 6),
            smoothness_cost=round(smoothness_cost, 6),
            final_pole_angle_rad=round(final_angle, 6),
            max_abs_pole_angle_rad=round(max_angle, 6),
            rms_pole_angle_rad=round(rms_angle, 6),
            final_cart_position_m=round(final_cart, 6),
            max_abs_cart_position_m=round(max_cart, 6),
            final_minimum_phase_output_m=round(final_output, 6),
            rms_minimum_phase_output_m=round(rms_output, 6),
        )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
