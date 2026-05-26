# Differential Drive: Lidar Maze Navigation

## Identity

- `challenge_id`: `diff_drive_lidar_maze`
- Status: `runnable`
- Difficulty: intermediate
- Category: mobile robot navigation
- Robot/fixture: `differential_drive_cube_v1`

## Objective

Write a `RobotPolicy` that drives a 12 inch-class differential-drive robot from the maze entrance to the exit target without touching the maze baffles or arena walls. Policies should use the 360-degree lidar scan to maintain clearance while following the corridor.

## Composition Decision

- Reuse: existing differential-drive policy API, runner, MuJoCo world builder, and Rerun export.
- Extend: add a seeded `lidar_maze_navigation` scenario to the existing runner.
- New infrastructure: none.
- Reason: the task is planar wheel-velocity navigation with the same replay, collision, lidar, and scoring shape as existing differential-drive navigation challenges.

## Embodiment And Environment

The robot uses the existing `differential_drive_cube_v1` primitive with a 360-degree top lidar, bounded arena walls, and circular collision geometry. The maze is approximated as three staggered internal baffle walls built from close-spaced cylindrical posts. The seed jitters the start pose, target pose, and baffle posts while preserving the top-bottom-top serpentine route.

## Interface Contract

- Injected class: `DifferentialDrive`
- Base class: `MobileRobot`
- Public constants: `wheel_base_m`, `wheel_radius_m`, `max_wheel_velocity_mps`, `dt`, `max_steps`, `seed`
- Sensors: `get_pose()`, `get_target()`, `route()`, `lidar()`
- Actuators: `set_wheel_velocity(left_mps, right_mps)`
- Hidden implementation: MuJoCo model construction, collision checks, scoring, seed-specific wall jitter, and replay export.

## Stochastic Contract

The public run is deterministic for a fixed seed. The seed randomizes start pose, target pose, and wall-post jitter. Lidar ranges are exact analytic ranges against arena bounds and circular baffle posts in the public runner; hidden variants may add bounded range noise or dropout if the catalog contract is updated.

## Scenario Generation

The default evaluation seed is `17`. The robot starts near the lower-left maze entrance with a small heading perturbation. The target is near the upper-right maze exit. Three staggered baffle walls create alternating top and bottom openings. Route waypoints mark the nominal corridor centerline but policies are expected to use lidar for local clearance.

## Success And Scoring

- Success: final robot center enters the target disk with zero collisions.
- Timeout: `32.0` seconds.
- Collision/safety penalties: any baffle or arena-wall collision fails the success condition and reduces safety score.
- Score components: reach success, time remaining, collision safety, wheel-command smoothness, and wheel-effort energy.
- Metrics returned: standard navigation metrics, including status, score, elapsed time, final distance, collision count, path length, energy, and smoothness cost.

## MuJoCo And Rerun Output

- MuJoCo cameras: `overview` and `front_camera` when MuJoCo is installed.
- Render cadence: `0.35` seconds by default.
- Rerun static scene streams: arena bounds, target disk, baffle posts, route, robot body, and trajectory.
- Rerun time-series streams: robot transforms, wheel commands, lidar point clouds, and MuJoCo render frames.
- Replay metadata: seed, timestep, challenge mode, MuJoCo backend, render cameras, render frame count, robot dimensions, route waypoints, lidar scan count, and arena bounds.

## Demo Policy

The sample policy at `samples/policies/lidar_maze_follower.py` follows the route waypoints while using lidar sectors to slow for frontal obstacles and bias steering away from nearby side walls. It is a reference sample for local validation; frontend starter code should remain a smaller API exercise when this environment is surfaced in the app.

## Validation Plan

- Backend tests: `uv run pytest`
- Frontend build: covered in the RoboRank app after the package commit is pinned.
- Browser run: covered in the RoboRank app after the package commit is pinned.
- MuJoCo verification: local runs should expose render frames when the optional visualization dependencies are installed.
- Rerun verification: local runs should write a `.rrd` artifact unless `ROBORANK_DISABLE_RERUN_EXPORT=1` is set or `rerun-sdk` is unavailable.
