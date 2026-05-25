# Quadrotor: Gate Sequence

## Identity

- `challenge_id`: `quadrotor_gate_sequence`
- Status: `runnable`
- Difficulty: Advanced
- Category: Aerial robotics
- Robot/fixture: `quadrotor_x500_v1`

## Objective

Write a `RobotPolicy` that flies a quadrotor through a seeded sequence of three vertical gates in order while staying inside the flight volume and within attitude limits.

## Composition Decision

- Reuse: catalog models, policy loader, run response, MuJoCo render frames, and Rerun artifact export.
- Extend: shared robot specs, policy API, frontend challenge composer, replay normalization, and runner routing.
- New infrastructure: a quadrotor gate runner and MuJoCo visualization world.
- Reason: this embodiment cannot be represented honestly by the planar differential-drive runner.

## Embodiment And Environment

The robot is a small X-frame quadrotor with analytic rigid-body translational dynamics. The flight course uses a world ENU frame: `+x` is course-forward, `+y` is left, and `+z` is up. The body frame moves with the drone: `+x` is nose-forward, `+y` is left, and `+z` is up. Gates are vertical rectangles with a yawed normal; a gate is passed from the negative-normal side to the positive-normal side.

## Interface Contract

- Injected class: `Quadrotor`
- Base class: `AerialRobot`
- Public constants: `hover_power`, `max_power`, `max_body_rate_radps`, `max_tilt_rad`, `dt`, `max_steps`, `seed`
- Sensors: `get_pose() -> Pose3d`, `get_altitude() -> float`, `get_next_gate() -> Gate3d`, `gates_completed() -> int`, `gate_count() -> int`
- Actuators: `set_body_rate_and_power(roll_rate_radps, pitch_rate_radps, yaw_rate_radps, power) -> None`
- Hidden implementation: dynamics integration, seeded gate placement, collision checks, scoring, rendering, and Rerun export.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Start yaw, start altitude, gate centers, and small gate-yaw offsets vary by seed. Public pose and altitude are exact in this runner; hidden variants may add declared estimator noise, latency, and wind gusts without changing the frame convention.

## Scenario Generation

Each seed creates three gates down the `+x` course with lateral and altitude variation. The quadrotor starts near `(0, 0, 1)` with zero velocity and small yaw offset. Episode length is 15 seconds at a 20 ms policy step.

## Success And Scoring

- Success: all three gates passed in order.
- Timeout: not all gates completed within 15 seconds.
- Collision/safety penalties: gate-frame misses, flight-volume exits, and tilt above 55 degrees fail the run.
- Score components: success, elapsed time, partial gate progress, safety margin, smoothness, and power use.
- Metrics returned: score, status, elapsed time, distance to next gate, collisions, path length, energy, smoothness, gates completed, gate count, and max attitude.

## MuJoCo And Rerun Output

- MuJoCo cameras: fixed `overview` course camera and body-mounted `front_camera`.
- Render cadence: 10 Hz plus terminal events.
- Rerun static scene streams: flight volume, gates, gate normals, drone body, trajectory, and summary text.
- Rerun time-series streams: drone pose, altitude, attitude, speed, body-rate commands, collective power, and encoded MuJoCo frames.
- Replay metadata: seed, timestep, render backend, frame counts, flight volume, gate sequence, gate pass times, robot dimensions, and frame definitions.

## Demo Policy

The frontend starter policy reads pose, gate progress, and next-gate metadata, then hovers, moves forward roughly one meter, brakes, and hovers again. It is meant to teach altitude hold and body-rate/power control without solving the gate sequence.

## Validation Plan

- Backend tests: catalog detail, API run with sample gate tracker, deterministic runner output, and bad action validation.
- Frontend build: generated API stub, starter code, playable status, replay normalization, and metric rendering.
- Browser run: open `/problems/quadrotor_gate_sequence`, run the starter or sample code, and confirm backend execution.
- MuJoCo verification: render frames are returned for `overview` and `front_camera`.
- Rerun verification: a `.rrd` artifact is written and linked in the run response.
