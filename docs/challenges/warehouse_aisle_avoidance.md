# Warehouse Aisle Avoidance

## Identity

- `challenge_id`: `warehouse_aisle_avoidance`
- Status: `runnable`
- Difficulty: Intermediate
- Category: Planning and control
- Robot/fixture: 12 inch-class differential-drive mobile robot in a warehouse aisle

## Objective

Submitted `RobotPolicy` code must command wheel velocities that drive from the aisle entrance to the aisle exit while avoiding rack posts, pallet intrusions, and arena bounds.

## Composition Decision

- Reuse: existing differential-drive policy API, MuJoCo world builder, collision checks, lidar, scoring, and Rerun exporter.
- Extend: the differential-drive runtime now exposes `route() -> tuple[Pose2d, ...]` so policies can follow seeded aisle waypoints.
- New infrastructure: none beyond the challenge catalog entry, scenario generator branch, frontend problem contract, sample policy, and tests.
- Reason: the task is planar navigation with a differential-drive base, wheel velocity commands, static circular obstacles, and the same replay requirements as the existing target-reaching challenge.

## Embodiment And Environment

The robot uses the existing `differential_drive_cube_v1` primitive. The arena is a bounded warehouse aisle with paired rack posts on both sides, two pallet intrusions into the aisle, and a staging cone. Obstacles are modeled as circular collision geometry in the public runner.

## Interface Contract

- Injected class: `DifferentialDrive`
- Base class: `MobileRobot`
- Public constants: `wheel_base_m`, `max_wheel_velocity_mps`
- Sensors: `get_pose()`, `get_target()`, `route()`, `obstacles()`, `lidar()`
- Actuators: `set_wheel_velocity(left_mps, right_mps)`
- Hidden implementation: scenario generation, MuJoCo stepping, collision checks, rendering, and Rerun export.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Start pose, target pose, rack-post offsets, and pallet intrusion positions vary by seed. Public lidar is deterministic and range-limited; hidden variants must declare any added noise, latency, dropout, or bias in the problem contract.

## Scenario Generation

The seeded generator places the robot near the aisle entrance, the target near the exit, rack posts along both aisle sides, and two pallet intrusions requiring an S-shaped path. It also provides route waypoints through the available aisle corridor. Episode length is 26 seconds at 20 Hz.

## Success And Scoring

- Success: final distance to aisle exit is at most `0.24 m` and `collision_count` is zero.
- Timeout: `26 sec`.
- Collision/safety penalties: obstacle and bounds contacts end the episode and reduce safety score.
- Score components: success/progress, time remaining, safety, smoothness, and wheel effort.
- Metrics returned: score, success, status, elapsed time, distance to target, collision count, path length, energy, and smoothness cost.

## MuJoCo And Rerun Output

- MuJoCo cameras: `overview` and `front_camera`.
- Render cadence: every `0.1 sec` when MuJoCo rendering is available.
- Rerun static scene streams: arena bounds, target, obstacles, route, robot box, front axis, and trajectory.
- Rerun time-series streams: robot pose transform, wheel controls, lidar points, and encoded MuJoCo images.
- Replay metadata: seed, timestep, render cameras, render frame count, route waypoints, arena bounds, robot dimensions, lidar scan count, and any render/export errors.

## Demo Policy

The frontend starter reads pose, route waypoints, and lidar, points gently toward the first route waypoint, and stops or turns when the front lidar sector is close. It demonstrates `route()`, `lidar()`, and wheel commands without acting as a route follower for the full aisle.

## Validation Plan

- Backend tests: catalog detail, sample policy run, browser source payload, bad policy validation, deterministic runner behavior.
- Frontend build: TypeScript and Vite production build.
- Browser run: open `/problems/warehouse_aisle_avoidance`, verify the generated API stub, run the starter policy, and inspect replay panes.
- MuJoCo verification: run response includes render frames from both cameras when optional dependencies are available.
- Rerun verification: run response includes a `.rrd` artifact or reports a Rerun export error in metadata.
