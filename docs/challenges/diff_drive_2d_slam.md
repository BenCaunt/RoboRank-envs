# Differential Drive: 2D SLAM

## Identity

- `challenge_id`: `diff_drive_2d_slam`
- Status: `runnable`
- Difficulty: advanced
- Category: State estimation
- Robot/fixture: `differential_drive_cube_v1`

## Objective

Submitted `RobotPolicy` code must build a field-frame `Pose2d` estimate and occupied-point map from cumulative wheel encoder ticks, biased gyro yaw rate, and noisy 360-degree lidar. The final map and final pose are scored after a long loop that revisits the start pose.

## Composition Decision

- Reuse: differential-drive robot primitive, lidar ray casting, MuJoCo world builder, Rerun export, odometry trace models.
- Extend: add `DifferentialDriveSlam`, a `diff_drive_slam` runner, map-point replay, and SLAM-specific scoring.
- New infrastructure: final occupied-point map scoring with Chamfer distance, coverage, and false-positive ratio.
- Reason: the existing odometry and AprilTag tasks score pose estimates only; this challenge must make dead-reckoned scan accumulation fail and reward loop closure or filtering.

## Embodiment And Environment

The robot is the existing 12 inch-class differential-drive cube with a top-mounted 360-degree planar lidar. A hidden controller drives a closed Lissajous-style path inside a rectangular arena with walls and seeded circular landmark posts. The course length is about 12 m, which is long enough for encoder scale error and gyro bias to smear a map if scans are transformed only by raw odometry.

## Interface Contract

- Injected class: `DifferentialDriveSlam`
- Base class: `MobileRobot`
- Public constants: `wheel_base_m`, `wheel_radius_m`, `ticks_per_rev`, `dt`, `lidar_max_range_m`, `max_map_points`
- Sensors: `get_encoder_values()`, `gyro()`, `lidar()`, `lidar_angles()`
- Actuators/submissions: `submit_slam(pose: Pose2d, map_points: list[MapPoint2d])`
- Hidden implementation: ground-truth pose, trajectory generation, landmark layout, sensor bias/noise, MuJoCo rendering, Rerun export, and map scoring.

## Stochastic Contract

Public runs are deterministic for a fixed evaluation seed. Encoder ticks include seeded left/right scale error, gyro readings include seeded bias and Gaussian noise, lidar ranges include seeded Gaussian range noise and 5 mm quantization, and landmark positions include small seeded jitter. No solved pose, solved map, or reset hook is exposed.

## Scenario Generation

The default seed is `13`. The robot starts at `x=0, y=0, yaw=0`, follows an 840-step closed loop at `dt=0.05`, and returns to the start pose near the final timestep. Arena bounds are `[-3.0, 3.0] x [-2.35, 2.35]` meters. Landmark posts are placed near the route but outside collision clearance.

## Success And Scoring

- Success: final pose error <= `0.18 m`, final yaw error <= `8 deg`, map Chamfer error <= `0.08 m`, map coverage ratio >= `0.72`, map false-positive ratio <= `0.20`, and no collisions.
- Timeout: 42 seconds simulated time.
- Collision/safety penalties: any collision fails success; no collisions add a small safety bonus.
- Score components: final loop-closure pose, final yaw, map Chamfer accuracy, visible-surface coverage, map precision, and safety.
- Metrics returned: `metric_kind=slam`, final/mean/max pose error, final/mean yaw error, path excitation, map Chamfer error, coverage ratio, false-positive ratio, and map point count.

## MuJoCo And Rerun Output

- MuJoCo cameras: existing mobile robot `overview` and `front_camera` cameras.
- Render cadence: default 0.5 seconds.
- Rerun static scene streams: arena bounds, target loop-closure disk, landmark posts, route, robot shape, ground-truth trajectory, submitted pose trajectory, and submitted SLAM map points.
- Rerun time-series streams: robot transform, lidar point clouds, encoder ticks, gyro samples, submitted pose error, controls for the hidden trajectory, and MuJoCo frames.
- Replay metadata: seed, timestep, sensor bias/noise parameters, map point counts, render backend, route waypoints, and arena bounds.

## Demo Policy

The frontend starter reads encoders, gyro, lidar ranges, and lidar angles, then submits a small local scan sketch at the origin. It exercises the public API and replay path but does not integrate odometry, close loops, or produce a passable map.

## Validation Plan

- Backend tests: catalog detail, successful loop-closure policy, failing odometry-only baseline, replay metadata, lidar/render/artifact presence, and missing-submission validation.
- Frontend build: ensure the generated `DifferentialDriveSlam` stub and starter compile.
- Browser run: open `/problems/diff_drive_2d_slam`, run the starter, confirm backend execution, MuJoCo frames, and Rerun artifact.
- MuJoCo verification: render frames should show the closed-loop arena and robot.
- Rerun verification: recording should include submitted SLAM map points and pose trajectory.
