# Kalman Acceleration Estimation

## Identity

- `challenge_id`: `kalman_acceleration_estimation`
- Status: `runnable`
- Difficulty: Intermediate
- Category: State estimation
- Robot/fixture: `profiled_cart_1d_v1`

## Objective

Write `RobotPolicy.step(cart)` to estimate the current acceleration of a hidden 1D cart from a noisy distance-to-wall sensor. The runner drives the cart back and forth on a seeded smooth trajectory. The submitted policy must call `cart.submit_acceleration(acceleration_mps2)` every timestep.

## Composition Decision

- Reuse: profiled 1D cart robot primitive, policy loader, runner dispatch, MuJoCo render response shape, and Rerun artifact plumbing.
- Extend: policy API with `AccelerationEstimator1D`, replay traces with acceleration estimate samples, and metrics with RMSE, correlation, and phase lag.
- New infrastructure: dedicated distance-cart runner and MuJoCo wall-distance visual world.
- Reason: the embodiment is the existing 1D cart, but the exposed task is estimation from a noisy sensor rather than commanding motion.

## Embodiment And Environment

The cart moves on a bounded 1D rail. A wall is fixed near the positive end of the track. The hidden trajectory is a seeded sum of smooth sinusoids that produces repeated forward and backward motion with changing acceleration. The policy never sees true position, velocity, or acceleration.

## Interface Contract

- Injected class: `AccelerationEstimator1D`
- Base class: `ControlSystem`
- Public constants: `wall_position_m`, `track_half_width_m`, `distance_noise_std_m`, `distance_quantization_m`, `dt`, `max_steps`, `seed`
- Sensor: `distance_to_wall() -> float`
- Output: `submit_acceleration(acceleration_mps2)`
- Hidden implementation: true trajectory, distance bias, Gaussian measurement noise draw, scoring, rendering, and Rerun export.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Distance measurements include a seeded constant bias, zero-mean Gaussian noise with 0.035 m standard deviation, and 0.002 m quantization. The noise is intentionally large enough that raw second differences of measured distance produce unusable acceleration estimates.

## Scenario Generation

The seed controls the sinusoid amplitudes, frequencies, phases, center offset, and distance-sensor bias. The wall position is fixed at 3.7 m. The default timestep is 0.02 s and the episode length is 14.0 s.

## Success And Scoring

- Success: RMS acceleration error <= 1.0 m/s^2, mean absolute error <= 0.82 m/s^2, phase lag <= 0.28 s, and acceleration correlation >= 0.84 after a 0.8 s estimator warmup.
- Timeout: the full fixed-length trace is always evaluated.
- Score components: RMS error, mean absolute error, phase lag, correlation, and a small bonus for beating raw double differentiation and a trailing moving-average baseline.
- Metrics returned: `acceleration_rmse_mps2`, `mean_abs_acceleration_error_mps2`, `phase_lag_sec`, `acceleration_correlation`, `derivative_baseline_rmse_mps2`, and `moving_average_baseline_rmse_mps2`.

## MuJoCo And Rerun Output

- MuJoCo cameras: `overview` and `side`
- Render cadence: 0.1 s
- Rerun static scene streams: rail, wall, cart path, and summary text
- Rerun time-series streams: true position, velocity, true acceleration, estimated acceleration, acceleration error, measured distance, measured position, and MuJoCo frames
- Replay metadata: timestep, wall position, sensor noise model, warmup duration, render backend, render count, and artifact status

## Demo Policy

The frontend starter reads `distance_to_wall()` and submits a placeholder acceleration estimate. It demonstrates the public API without implementing the expected PVA Kalman filter.

## Validation Plan

- Backend tests: catalog detail, sample Kalman policy success, raw double-differentiation failure, replay metadata, and acceleration estimate trace presence
- Frontend build: generated stub and starter policy compile in the app
- Browser run: open `/problems/kalman_acceleration_estimation`, run the starter, and inspect MuJoCo/Rerun panes
- Rerun verification: confirm a `.rrd` artifact is produced when Rerun is installed
