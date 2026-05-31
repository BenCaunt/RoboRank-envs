from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from roborank_envs.catalog import KALMAN_ACCELERATION_ESTIMATION
from roborank_envs.models import (
    AccelerationEstimateSample,
    ChallengeSpec,
    ReplayTrace,
    RenderFrame,
    RunResult,
    ScoreMetrics,
)
from roborank_envs.policy_api import AccelerationEstimator1D as AccelerationEstimatorApi
from roborank_envs.policy_api import RobotPolicyProtocol
from roborank_envs.simulation.distance_cart_world import DistanceCartWorld
from roborank_envs.simulation.rendering import configured_render_interval_sec
from roborank_envs.simulation.rerun_export import write_rerun_recording


@dataclass(frozen=True)
class SinusoidComponent:
    amplitude_m: float
    frequency_hz: float
    phase_rad: float


@dataclass(frozen=True)
class AccelerationScenario:
    center_m: float
    components: tuple[SinusoidComponent, ...]
    wall_position_m: float
    sensor_bias_m: float


@dataclass(frozen=True)
class DistanceCartState:
    position_m: float
    velocity_mps: float
    acceleration_mps2: float


class SimulationError(RuntimeError):
    pass


class AccelerationEstimationRunner:
    def __init__(self, challenge: ChallengeSpec = KALMAN_ACCELERATION_ESTIMATION) -> None:
        self.challenge = challenge
        self.dt = float(challenge.defaults["dt_sec"])
        self.default_max_steps = int(challenge.defaults["max_steps"])
        self.render_interval_sec = configured_render_interval_sec(
            float(challenge.defaults.get("render_interval_sec", 0.1))
        )
        if challenge.robot.type != "profiled_cart_1d":
            raise ValueError("AccelerationEstimationRunner requires a profiled_cart_1d challenge.")

    def run(self, policy: RobotPolicyProtocol, *, seed: int = 0, max_steps: int | None = None) -> RunResult:
        random.seed(seed)
        np.random.seed(seed)

        cart = self.challenge.robot
        max_step_count = max_steps or self.default_max_steps
        scenario = self._scenario(seed)
        noise_rng = np.random.default_rng(seed + 2027)
        world = DistanceCartWorld.create(cart=cart, wall_position_m=scenario.wall_position_m)
        policy_cart = AccelerationEstimatorApi(
            wall_position_m=scenario.wall_position_m,
            track_half_width_m=cart.track_half_width_m,
            distance_noise_std_m=float(self.challenge.defaults["distance_noise_std_m"]),
            distance_quantization_m=float(self.challenge.defaults["distance_quantization_m"]),
            dt=self.dt,
            max_steps=max_step_count,
            seed=seed,
        )

        samples: list[AccelerationEstimateSample] = []
        render_frames: list[RenderFrame] = []
        measured_distances: list[float] = []
        render_error: str | None = None
        rerun_error: str | None = None

        try:
            render_stride = max(1, round(self.render_interval_sec / self.dt))

            for step in range(max_step_count):
                t = step * self.dt
                state = self._trajectory_state(t=t, scenario=scenario)
                measured_distance = self._distance_measurement(
                    state=state,
                    scenario=scenario,
                    noise_rng=noise_rng,
                )
                measured_distances.append(measured_distance)

                world.update(cart_position_m=state.position_m)
                if step % render_stride == 0:
                    render_error = self._append_render_frames(
                        world=world,
                        frames=render_frames,
                        t=t,
                        frame_index=step,
                        render_error=render_error,
                    )

                policy_cart._update(time=round(t, 6), distance_m=measured_distance)
                estimate = self._call_policy_step(policy, policy_cart)
                samples.append(
                    self._sample(
                        t=t,
                        frame_index=step,
                        state=state,
                        scenario=scenario,
                        measured_distance=measured_distance,
                        estimate=estimate,
                    )
                )
        finally:
            world.close()

        elapsed = max_step_count * self.dt
        metrics = self._score(
            elapsed=elapsed,
            samples=samples,
            measured_distances=measured_distances,
            scenario=scenario,
        )
        replay = ReplayTrace(
            acceleration_estimates=samples,
            render_frames=render_frames,
            metadata={
                "dt_sec": self.dt,
                "max_steps": max_step_count,
                "seed": seed,
                "runner": "kalman_acceleration_estimation",
                "challenge_mode": "kalman_acceleration_estimation",
                "mujoco_backend": world.backend_name,
                "simulation_mode": "analytic_distance_cart_with_mujoco_render"
                if world.available
                else "analytic_distance_cart_fallback",
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
                "wall_position_m": scenario.wall_position_m,
                "track_half_width_m": cart.track_half_width_m,
                "distance_noise_std_m": self.challenge.defaults["distance_noise_std_m"],
                "distance_quantization_m": self.challenge.defaults["distance_quantization_m"],
                "sensor_bias_m": round(scenario.sensor_bias_m, 6),
                "warmup_sec": self.challenge.defaults.get("warmup_sec", 0.0),
                "trajectory_components": [
                    {
                        "amplitude_m": component.amplitude_m,
                        "frequency_hz": component.frequency_hz,
                    }
                    for component in scenario.components
                ],
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

    def _scenario(self, seed: int) -> AccelerationScenario:
        rng = random.Random(seed)
        amplitudes = [
            rng.uniform(0.72, 0.92),
            rng.uniform(0.28, 0.42),
            rng.uniform(0.10, 0.18),
        ]
        frequencies = [
            rng.uniform(0.13, 0.18),
            rng.uniform(0.25, 0.34),
            rng.uniform(0.39, 0.48),
        ]
        phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(3)]
        return AccelerationScenario(
            center_m=rng.uniform(-0.15, 0.15),
            components=tuple(
                SinusoidComponent(
                    amplitude_m=round(amplitude, 6),
                    frequency_hz=round(frequency, 6),
                    phase_rad=phase,
                )
                for amplitude, frequency, phase in zip(amplitudes, frequencies, phases, strict=True)
            ),
            wall_position_m=float(self.challenge.defaults["wall_position_m"]),
            sensor_bias_m=rng.uniform(
                -float(self.challenge.defaults["distance_bias_range_m"]),
                float(self.challenge.defaults["distance_bias_range_m"]),
            ),
        )

    def _trajectory_state(self, *, t: float, scenario: AccelerationScenario) -> DistanceCartState:
        position = scenario.center_m
        velocity = 0.0
        acceleration = 0.0
        for component in scenario.components:
            omega = 2.0 * math.pi * component.frequency_hz
            phase = omega * t + component.phase_rad
            position += component.amplitude_m * math.sin(phase)
            velocity += component.amplitude_m * omega * math.cos(phase)
            acceleration -= component.amplitude_m * omega * omega * math.sin(phase)
        return DistanceCartState(
            position_m=position,
            velocity_mps=velocity,
            acceleration_mps2=acceleration,
        )

    def _distance_measurement(
        self,
        *,
        state: DistanceCartState,
        scenario: AccelerationScenario,
        noise_rng: np.random.Generator,
    ) -> float:
        noise_std = float(self.challenge.defaults["distance_noise_std_m"])
        quantization = float(self.challenge.defaults["distance_quantization_m"])
        measured = scenario.wall_position_m - state.position_m + scenario.sensor_bias_m
        measured += float(noise_rng.normal(0.0, noise_std))
        if quantization > 0.0:
            measured = round(measured / quantization) * quantization
        return round(max(0.0, measured), 6)

    def _call_policy_step(self, policy: RobotPolicyProtocol, cart: AccelerationEstimatorApi) -> float:
        try:
            cart._clear_submission()
            step_result = policy.step(cart)
            if step_result is not None:
                raise SimulationError(
                    "RobotPolicy.step(cart) should call cart.submit_acceleration(acceleration_mps2), not return an action."
                )
            return cart._consume_acceleration_estimate()
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

    def _sample(
        self,
        *,
        t: float,
        frame_index: int,
        state: DistanceCartState,
        scenario: AccelerationScenario,
        measured_distance: float,
        estimate: float,
    ) -> AccelerationEstimateSample:
        measured_position = scenario.wall_position_m - measured_distance
        error = estimate - state.acceleration_mps2
        return AccelerationEstimateSample(
            t=round(t, 6),
            frame_index=frame_index,
            position_m=round(state.position_m, 6),
            velocity_mps=round(state.velocity_mps, 6),
            acceleration_mps2=round(state.acceleration_mps2, 6),
            wall_position_m=round(scenario.wall_position_m, 6),
            measured_distance_m=round(measured_distance, 6),
            measured_position_m=round(measured_position, 6),
            estimated_acceleration_mps2=round(estimate, 6),
            acceleration_error_mps2=round(error, 6),
        )

    def _append_render_frames(
        self,
        *,
        world: DistanceCartWorld,
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
        samples: list[AccelerationEstimateSample],
        measured_distances: list[float],
        scenario: AccelerationScenario,
    ) -> ScoreMetrics:
        warmup_steps = min(
            max(0, int(round(float(self.challenge.defaults.get("warmup_sec", 0.0)) / self.dt))),
            max(0, len(samples) - 3),
        )
        actual = np.asarray([sample.acceleration_mps2 for sample in samples], dtype=float)
        estimated = np.asarray([sample.estimated_acceleration_mps2 for sample in samples], dtype=float)
        active_actual = actual[warmup_steps:]
        active_estimated = estimated[warmup_steps:]
        errors = active_estimated - active_actual
        rmse = math.sqrt(float(np.mean(errors * errors))) if len(errors) else 0.0
        mae = float(np.mean(np.abs(errors))) if len(errors) else 0.0
        final_error = abs(float(errors[-1])) if len(errors) else 0.0
        phase_lag, correlation = _phase_lag_seconds(
            truth=active_actual,
            estimate=active_estimated,
            dt=self.dt,
            max_lag_sec=float(self.challenge.defaults.get("phase_lag_search_sec", 0.5)),
        )
        derivative_baseline = _double_difference_baseline(
            measured_distances=measured_distances,
            wall_position_m=scenario.wall_position_m,
            dt=self.dt,
        )
        moving_average_baseline = _trailing_average_double_difference_baseline(
            measured_distances=measured_distances,
            wall_position_m=scenario.wall_position_m,
            dt=self.dt,
            window_steps=int(self.challenge.defaults.get("moving_average_baseline_window_steps", 31)),
        )
        derivative_rmse = _rmse(derivative_baseline[warmup_steps:], active_actual)
        moving_average_rmse = _rmse(moving_average_baseline[warmup_steps:], active_actual)

        rms_limit = float(self.challenge.success_conditions["rms_acceleration_error_mps2"])
        mae_limit = float(self.challenge.success_conditions["mean_abs_acceleration_error_mps2"])
        lag_limit = float(self.challenge.success_conditions["phase_lag_sec"])
        correlation_limit = float(self.challenge.success_conditions["min_acceleration_correlation"])

        success = (
            rmse <= rms_limit
            and mae <= mae_limit
            and phase_lag <= lag_limit
            and correlation >= correlation_limit
        )
        rms_points = 40.0 * _score_ratio(rmse, rms_limit)
        mae_points = 22.0 * _score_ratio(mae, mae_limit)
        lag_points = 18.0 * _score_ratio(phase_lag, lag_limit)
        correlation_points = 15.0 * _clamp01((correlation - 0.35) / max(correlation_limit - 0.35, 1e-9))
        baseline_points = 5.0 if rmse < min(derivative_rmse, moving_average_rmse) else 0.0
        total = min(100.0, rms_points + mae_points + lag_points + correlation_points + baseline_points)

        return ScoreMetrics(
            metric_kind="acceleration_estimation",
            score=round(total, 3),
            success=success,
            status="success" if success else "accuracy_error",
            elapsed_sec=round(elapsed, 6),
            distance_to_target_m=round(rmse, 6),
            collision_count=0,
            path_length_m=round(
                sum(
                    abs(samples[index].position_m - samples[index - 1].position_m)
                    for index in range(1, len(samples))
                ),
                6,
            ),
            energy_used=0.0,
            smoothness_cost=round(
                sum(
                    (samples[index].estimated_acceleration_mps2 - samples[index - 1].estimated_acceleration_mps2) ** 2
                    * self.dt
                    for index in range(1, len(samples))
                ),
                6,
            ),
            final_position_error_m=round(rmse, 6),
            mean_position_error_m=round(mae, 6),
            final_acceleration_error_mps2=round(final_error, 6),
            acceleration_rmse_mps2=round(rmse, 6),
            mean_abs_acceleration_error_mps2=round(mae, 6),
            phase_lag_sec=round(phase_lag, 6),
            acceleration_correlation=round(correlation, 6),
            derivative_baseline_rmse_mps2=round(derivative_rmse, 6),
            moving_average_baseline_rmse_mps2=round(moving_average_rmse, 6),
        )


def _double_difference_baseline(*, measured_distances: list[float], wall_position_m: float, dt: float) -> np.ndarray:
    position = np.asarray([wall_position_m - distance for distance in measured_distances], dtype=float)
    acceleration = np.zeros_like(position)
    if len(position) >= 3:
        acceleration[2:] = (position[2:] - 2.0 * position[1:-1] + position[:-2]) / (dt * dt)
    return acceleration


def _trailing_average_double_difference_baseline(
    *,
    measured_distances: list[float],
    wall_position_m: float,
    dt: float,
    window_steps: int,
) -> np.ndarray:
    position = np.asarray([wall_position_m - distance for distance in measured_distances], dtype=float)
    smoothed = np.zeros_like(position)
    window = max(1, int(window_steps))
    for index in range(len(position)):
        start = max(0, index - window + 1)
        smoothed[index] = float(np.mean(position[start : index + 1]))
    acceleration = np.zeros_like(position)
    if len(position) >= 3:
        acceleration[2:] = (smoothed[2:] - 2.0 * smoothed[1:-1] + smoothed[:-2]) / (dt * dt)
    return acceleration


def _phase_lag_seconds(
    *,
    truth: np.ndarray,
    estimate: np.ndarray,
    dt: float,
    max_lag_sec: float,
) -> tuple[float, float]:
    if len(truth) < 8 or len(estimate) < 8:
        return 0.0, 0.0
    truth_zero_mean = truth - float(np.mean(truth))
    estimate_zero_mean = estimate - float(np.mean(estimate))
    max_lag_steps = max(1, int(round(max_lag_sec / dt)))
    best_correlation = -1.0
    best_lag_steps = 0
    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        if lag_steps >= 0:
            truth_segment = truth_zero_mean[: len(truth_zero_mean) - lag_steps or None]
            estimate_segment = estimate_zero_mean[lag_steps:]
        else:
            truth_segment = truth_zero_mean[-lag_steps:]
            estimate_segment = estimate_zero_mean[: len(estimate_zero_mean) + lag_steps]
        if len(truth_segment) < 8:
            continue
        denominator = float(np.linalg.norm(truth_segment) * np.linalg.norm(estimate_segment))
        correlation = 0.0 if denominator <= 1e-12 else float(np.dot(truth_segment, estimate_segment) / denominator)
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag_steps = lag_steps
    return abs(best_lag_steps * dt), best_correlation


def _rmse(values: np.ndarray, truth: np.ndarray) -> float:
    if len(values) == 0 or len(truth) == 0:
        return 0.0
    length = min(len(values), len(truth))
    errors = values[:length] - truth[:length]
    return math.sqrt(float(np.mean(errors * errors)))


def _score_ratio(value: float, limit: float) -> float:
    if limit <= 0.0:
        return 0.0
    return _clamp01(1.25 - 0.5 * (value / limit))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
