# Differential Drive: Build Odometry

## Identity

- `challenge_id`: `diff_drive_odometry`
- Status: `runnable`
- Difficulty: `beginner`
- Category: State estimation
- Robot/fixture: `differential_drive_cube_v1`

## Objective

Submitted `RobotPolicy.step(robot)` code must integrate cumulative wheel encoder ticks and gyro yaw-rate samples into a `Pose2d` odometry estimate in the episode-start frame. The policy must submit an estimate every step while the environment drives a seeded calibration trajectory.

## Composition Decision

- Reuse: existing differential-drive robot primitive, MuJoCo world builder, render frame response, Rerun export, policy loader, and API route shape.
- Extend: differential-drive policy API with a lower-level odometry class and add an odometry runner variant with an internal trajectory generator.
- New infrastructure: replay samples for encoder ticks, gyro yaw rate, and submitted odometry estimates.
- Reason: the embodiment and planar dynamics match the current differential-drive stack, but solved-pose access must be removed and scoring must compare submitted estimates to hidden ground truth.

## Embodiment And Environment

The public course uses the 12 inch-class differential-drive cube in a bounded planar arena. The robot starts at `x=0`, `y=0`, `yaw=0` in the odometry frame. The runner drives a deterministic seeded wheel-velocity profile, uses kinematic differential-drive integration with a small seeded left/right slip scale, and uses MuJoCo for replay rendering.

## Interface Contract

- Injected class: `DifferentialDriveOdometry`
- Base class: `MobileRobot`
- Public constants: `wheel_base_m`, `wheel_radius_m`, `ticks_per_rev`, `dt`, `max_steps`, `seed`
- Sensors: `get_encoder_values() -> tuple[int, int]`, `gyro() -> float`
- Outputs: `submit_odometry(Pose2d)`
- Hidden implementation: ground-truth pose, seeded trajectory, slip scale, gyro bias/noise, render details, and scoring.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Encoder values are cumulative integer ticks quantized from the environment's internal wheel commands. Gyro samples include seeded constant bias plus Gaussian yaw-rate noise. A small seeded wheel-slip scale changes actual ground motion relative to encoder-derived wheel travel.

## Scenario Generation

The start pose is fixed in the odometry frame. Seed controls the internal trajectory profile, slip scale, gyro bias, and gyro noise stream. The episode runs for 14 seconds at 50 ms timesteps unless the robot leaves arena bounds.

## Success And Scoring

- Success: final position error <= 0.12 m, mean position error <= 0.08 m, final yaw error <= 5 degrees, and no collisions.
- Timeout: max step count reached without meeting the estimation tolerances.
- Collision/safety penalties: leaving arena bounds fails the run and reduces safety score.
- Score components: final position, mean trajectory position, yaw, and safety.
- Metrics returned: `final_position_error_m`, `mean_position_error_m`, `max_position_error_m`, `final_yaw_error_rad`, `mean_yaw_error_rad`, `excitation_distance_m`, `excitation_yaw_rad`.

## MuJoCo And Rerun Output

- MuJoCo cameras: `overview` and `front_camera`
- Render cadence: 10 Hz when MuJoCo is available
- Rerun static scene streams: arena bounds, route, actual trajectory, submitted odometry trajectory
- Rerun time-series streams: robot pose, wheel commands, encoder ticks, gyro yaw rate, odometry estimate, position error, yaw error, and MuJoCo images
- Replay metadata: seed, timestep, render backend, robot dimensions, encoder resolution, noise settings, slip scales, sample counts, and artifact status

## Demo Policy

The frontend starter reads encoder and gyro samples and submits a placeholder `Pose2d(0, 0, 0)`. It demonstrates the required `submit_odometry` call without providing an odometry solution.

## Validation Plan

- Backend tests: catalog detail, API run with sample policy, runner success, missing-estimate validation
- Frontend build: generated stub and starter policy compile in the app
- Browser run: open `/problems/diff_drive_odometry`, run the starter, confirm backend result
- MuJoCo verification: render frames appear in the replay response
- Rerun verification: `.rrd` artifact is attached and includes odometry sensor/error streams
