from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from roborank_envs.catalog import MOTOR_TORQUE_SCALE_CONTROL
from roborank_envs.models import (
    ChallengeSpec,
    MotorControlSample,
    MotorStateSample,
    ReplayTrace,
    RenderFrame,
    RunResult,
    ScoreMetrics,
)
from roborank_envs.policy_api import CurrentControlledMotor as CurrentControlledMotorApi
from roborank_envs.policy_api import RobotPolicyProtocol
from roborank_envs.simulation.motor_stand_world import MotorStandWorld
from roborank_envs.simulation.rendering import configured_render_interval_sec
from roborank_envs.simulation.rerun_export import write_rerun_recording


@dataclass(frozen=True)
class MotorScenario:
    target_force_n: float
    kt_nm_per_amp: float
    current_tau_sec: float
    scale_tau_sec: float
    current_bias_a: float
    scale_bias_n: float
    current_noise_std_a: float
    scale_noise_std_n: float
    contact_loss_n: float


@dataclass
class MotorPlantState:
    actual_current_a: float = 0.0
    scale_force_n: float = 0.0
    measured_current_a: float = 0.0
    measured_scale_force_n: float = 0.0
    shaft_angle_rad: float = -0.025
    shaft_tip_deflection_m: float = 0.0


class SimulationError(RuntimeError):
    pass


class MotorStandRunner:
    def __init__(self, challenge: ChallengeSpec = MOTOR_TORQUE_SCALE_CONTROL) -> None:
        self.challenge = challenge
        self.dt = float(challenge.defaults["dt_sec"])
        self.default_max_steps = int(challenge.defaults["max_steps"])
        self.render_interval_sec = configured_render_interval_sec(
            float(challenge.defaults.get("render_interval_sec", 0.1))
        )
        if challenge.robot.type != "motor_stand":
            raise ValueError("MotorStandRunner requires a motor_stand challenge.")

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        fixture = self.challenge.robot
        max_step_count = max_steps or self.default_max_steps
        rng = random.Random(seed)
        noise_rng = np.random.default_rng(seed)
        scenario = self._scenario(rng)
        world = MotorStandWorld.create(fixture=fixture)

        state = MotorPlantState()
        self._refresh_sensor_state(state=state, scenario=scenario, noise_rng=noise_rng)
        policy_motor = CurrentControlledMotorApi(
            shaft_length_m=fixture.shaft_length_m,
            kt_nm_per_amp=fixture.kt_nm_per_amp,
            max_current_a=fixture.max_current_a,
            target_force_n=scenario.target_force_n,
            dt=self.dt,
            max_steps=max_step_count,
            seed=seed,
        )

        motor_states: list[MotorStateSample] = []
        motor_controls: list[MotorControlSample] = []
        render_frames: list[RenderFrame] = []
        force_history: list[float] = []
        command_history: list[float] = []
        render_error: str | None = None
        rerun_error: str | None = None
        energy_used = 0.0
        smoothness_cost = 0.0
        last_command: float | None = None
        final_step = 0

        world.update(shaft_angle_rad=state.shaft_angle_rad, platen_deflection_m=state.shaft_tip_deflection_m)
        motor_states.append(
            self._state_sample(
                t=0.0,
                frame_index=0,
                command=0.0,
                state=state,
                scenario=scenario,
            )
        )
        force_history.append(state.measured_scale_force_n)

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
                policy_motor._update(
                    time=round(t, 6),
                    measured_current_a=state.measured_current_a,
                    scale_force_n=state.measured_scale_force_n,
                )
                command = self._call_policy_step(policy, policy_motor)
                motor_controls.append(MotorControlSample(t=round(t, 6), current_command_a=round(command, 6)))
                command_history.append(command)

                if last_command is not None:
                    smoothness_cost += ((command - last_command) ** 2) * self.dt
                last_command = command
                energy_used += (command**2) * self.dt

                self._integrate(state=state, command=command, fixture=fixture, scenario=scenario)
                self._refresh_sensor_state(state=state, scenario=scenario, noise_rng=noise_rng)
                final_step = step + 1
                t_next = final_step * self.dt
                force_history.append(state.measured_scale_force_n)

                world.update(shaft_angle_rad=state.shaft_angle_rad, platen_deflection_m=state.shaft_tip_deflection_m)
                motor_states.append(
                    self._state_sample(
                        t=t_next,
                        frame_index=final_step,
                        command=command,
                        state=state,
                        scenario=scenario,
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
        finally:
            world.close()

        elapsed = final_step * self.dt
        metrics = self._score(
            elapsed=elapsed,
            force_history=force_history,
            command_history=command_history,
            target_force=scenario.target_force_n,
            energy_used=energy_used,
            smoothness_cost=smoothness_cost,
        )
        replay = ReplayTrace(
            motor_controls=motor_controls,
            motor_states=motor_states,
            render_frames=render_frames,
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "mujoco_backend": world.backend_name,
                "simulation_mode": "motor_fixture_mujoco_visual" if world.available else "motor_fixture_analytic_fallback",
                "mujoco_timestep_sec": world.timestep if world.available else None,
                "mujoco_render_frames": len(render_frames),
                "mujoco_render_cameras": list(world.render_cameras),
                "mujoco_render_interval_sec": self.render_interval_sec if world.available else None,
                "mujoco_render_resolution": {
                    "width": world.render_width,
                    "height": world.render_height,
                },
                "mujoco_render_error": render_error,
                "robot_model": fixture.model,
                "target_force_n": scenario.target_force_n,
                "fixture_dimensions_m": {
                    "shaft_length": fixture.shaft_length_m,
                    "shaft_radius": fixture.shaft_radius_m,
                    "base_length": fixture.base_length_m,
                    "base_width": fixture.base_width_m,
                },
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

    def _scenario(self, rng: random.Random) -> MotorScenario:
        fixture = self.challenge.robot
        force_min, force_max = self.challenge.success_conditions["target_force_range_n"]
        return MotorScenario(
            target_force_n=round(rng.uniform(float(force_min), float(force_max)), 3),
            kt_nm_per_amp=fixture.kt_nm_per_amp * rng.uniform(0.985, 1.015),
            current_tau_sec=fixture.current_time_constant_sec * rng.uniform(0.9, 1.18),
            scale_tau_sec=fixture.scale_time_constant_sec * rng.uniform(0.9, 1.2),
            current_bias_a=rng.uniform(-0.018, 0.018),
            scale_bias_n=rng.uniform(-0.012, 0.012),
            current_noise_std_a=0.009,
            scale_noise_std_n=0.012,
            contact_loss_n=rng.uniform(0.0, 0.018),
        )

    def _call_policy_step(self, policy: RobotPolicyProtocol, motor: CurrentControlledMotorApi) -> float:
        try:
            motor._clear_command()
            step_result = policy.step(motor)
            if step_result is not None:
                raise SimulationError("RobotPolicy.step(motor) should call motor.set_current(amps), not return an action.")
            return motor._consume_current_command()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _integrate(
        self,
        *,
        state: MotorPlantState,
        command: float,
        fixture: object,
        scenario: MotorScenario,
    ) -> None:
        current_alpha = min(1.0, self.dt / max(scenario.current_tau_sec, 1e-6))
        state.actual_current_a += (command - state.actual_current_a) * current_alpha
        torque_nm = scenario.kt_nm_per_amp * state.actual_current_a
        ideal_force = max(0.0, torque_nm / self.challenge.robot.shaft_length_m - scenario.contact_loss_n)
        scale_alpha = min(1.0, self.dt / max(scenario.scale_tau_sec, 1e-6))
        state.scale_force_n += (ideal_force - state.scale_force_n) * scale_alpha
        state.scale_force_n = max(0.0, min(self.challenge.robot.max_force_n, state.scale_force_n))
        state.shaft_tip_deflection_m = state.scale_force_n / self.challenge.robot.scale_stiffness_n_per_m
        contact_angle = state.shaft_tip_deflection_m / max(self.challenge.robot.shaft_length_m, 1e-9)
        state.shaft_angle_rad = -0.045 - min(0.18, contact_angle + 0.02 * state.scale_force_n)

    def _refresh_sensor_state(
        self,
        *,
        state: MotorPlantState,
        scenario: MotorScenario,
        noise_rng: np.random.Generator,
    ) -> None:
        state.measured_current_a = state.actual_current_a + scenario.current_bias_a + float(
            noise_rng.normal(0.0, scenario.current_noise_std_a)
        )
        state.measured_scale_force_n = max(
            0.0,
            state.scale_force_n
            + scenario.scale_bias_n
            + 0.006 * math.sin(5.0 * state.scale_force_n)
            + float(noise_rng.normal(0.0, scenario.scale_noise_std_n)),
        )

    def _state_sample(
        self,
        *,
        t: float,
        frame_index: int,
        command: float,
        state: MotorPlantState,
        scenario: MotorScenario,
    ) -> MotorStateSample:
        return MotorStateSample(
            t=round(t, 6),
            frame_index=frame_index,
            current_command_a=round(command, 6),
            measured_current_a=round(state.measured_current_a, 6),
            motor_torque_nm=round(scenario.kt_nm_per_amp * state.actual_current_a, 6),
            shaft_angle_rad=round(state.shaft_angle_rad, 6),
            shaft_tip_deflection_m=round(state.shaft_tip_deflection_m, 6),
            scale_force_n=round(state.measured_scale_force_n, 6),
            target_force_n=round(scenario.target_force_n, 6),
            force_error_n=round(scenario.target_force_n - state.measured_scale_force_n, 6),
        )

    def _append_render_frames(
        self,
        *,
        world: MotorStandWorld,
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
        elapsed: float,
        force_history: list[float],
        command_history: list[float],
        target_force: float,
        energy_used: float,
        smoothness_cost: float,
    ) -> ScoreMetrics:
        tolerance = float(self.challenge.success_conditions["force_tolerance_n"])
        overshoot_limit = float(self.challenge.success_conditions["overshoot_pct"])
        settling_limit = float(self.challenge.success_conditions["settling_time_sec"])
        timeout = float(self.challenge.success_conditions["timeout_sec"])
        final_window = max(1, round(0.5 / self.dt))
        steady_forces = force_history[-final_window:]
        mean_force = sum(steady_forces) / len(steady_forces)
        final_error = abs(mean_force - target_force)
        mean_abs_error = sum(abs(force - target_force) for force in steady_forces) / len(steady_forces)
        peak_force = max(force_history) if force_history else 0.0
        overshoot_pct = max(0.0, (peak_force - target_force) / max(target_force, 1e-9) * 100.0)
        settling_time = self._settling_time(force_history=force_history, target_force=target_force, tolerance=tolerance)
        success = final_error <= tolerance and overshoot_pct <= overshoot_limit and settling_time <= settling_limit

        accuracy_points = 45.0 * max(0.0, 1.0 - final_error / max(3.0 * tolerance, 1e-9))
        settling_points = 20.0 * max(0.0, 1.0 - settling_time / timeout) if settling_time < math.inf else 0.0
        overshoot_points = 15.0 * max(0.0, 1.0 - overshoot_pct / max(2.0 * overshoot_limit, 1e-9))
        smoothness_points = max(0.0, 10.0 - 0.4 * smoothness_cost)
        energy_points = max(0.0, 10.0 - 0.06 * energy_used)
        total = min(100.0, accuracy_points + settling_points + overshoot_points + smoothness_points + energy_points)

        if success:
            status = "success"
        elif overshoot_pct > overshoot_limit:
            status = "overshoot"
        elif final_error > tolerance:
            status = "force_error"
        else:
            status = "timeout"

        return ScoreMetrics(
            metric_kind="force_control",
            score=round(total, 3),
            success=success,
            status=status,  # type: ignore[arg-type]
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(final_error, 6),
            collision_count=0,
            path_length_m=0.0,
            energy_used=round(energy_used, 6),
            smoothness_cost=round(smoothness_cost, 6),
            target_force_n=round(target_force, 6),
            final_force_error_n=round(final_error, 6),
            mean_abs_force_error_n=round(mean_abs_error, 6),
            settling_time_sec=round(settling_time, 6) if settling_time < math.inf else None,
            overshoot_pct=round(overshoot_pct, 6),
            peak_force_n=round(peak_force, 6),
        )

    def _settling_time(self, *, force_history: list[float], target_force: float, tolerance: float) -> float:
        required_tail = max(1, round(0.45 / self.dt))
        for index in range(len(force_history)):
            tail = force_history[index:]
            if len(tail) < required_tail:
                break
            if all(abs(force - target_force) <= tolerance for force in tail):
                return index * self.dt
        return math.inf
