# Trapezoidal Motion Profile

## Identity

- `challenge_id`: `trapezoidal_motion_profile`
- Status: `runnable`
- Difficulty: Intermediate
- Category: Motion planning
- Robot/fixture: `profiled_cart_1d_v1`

## Objective

Implement `RobotPolicy.step(cart)` as a trapezoidal motion-profile generator for a one-dimensional cart. The policy must command acceleration so the cart reaches a seeded target with near-zero final velocity.

## Embodiment And Environment

The environment is a 1D cart on a straight track. The analytic plant integrates acceleration commands with a fixed timestep, clips the simulated acceleration and velocity to keep the visual plant bounded, and records any requested limit violations for scoring.

## Interface Contract

- Injected class: `ProfiledCart1D`
- Base class: `ControlSystem`
- Public constants: `max_velocity_mps`, `max_acceleration_mps2`, `target_position_m`, `target_velocity_mps`, `dt`, `max_steps`, `seed`, `time`
- Sensors: `get_state() -> MotionState1D`, `get_target() -> MotionTarget1D`
- Actuators: `set_acceleration(acceleration_mps2: float) -> None`
- Hidden implementation: target generation, integration, limit checks, score calculation, MuJoCo rendering, and Rerun export.

## Scenario Generation

The cart starts at `x=0` with zero velocity. The target position is sampled from `target_distance_range_m`, with seeded sign when negative targets are enabled. The target velocity is zero.

## Success And Scoring

- Success: final position error <= 0.04 m, final velocity error <= 0.05 m/s, no acceleration or velocity limit violations, and finish time <= 1.35x the analytic time-optimal trapezoidal/triangular profile.
- Timeout: 8.0 seconds by default.
- Collision/safety penalties: no collisions; limit violations fail the run and remove limit-compliance score.
- Score components: 35 accuracy, 20 final velocity, 20 limit compliance, 20 time, 5 smoothness.
- Metrics returned: position error, velocity error, finish time, optimal time, max requested acceleration, acceleration-limit violations, velocity-limit violations.

## MuJoCo And Rerun Output

- MuJoCo cameras: `overview`, `side`
- Render cadence: 0.1 seconds by default.
- Rerun static scene streams: track, stops, target, and cart path.
- Rerun time-series streams: cart position, velocity, acceleration, target, acceleration command, applied acceleration, and limit flags.
- Replay metadata: runner id, timestep, target, limits, track width, optimal time, render settings, and artifact status.
