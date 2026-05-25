# Differential Drive: AprilTag State Estimation

## Identity

- `challenge_id`: `diff_drive_state_estimation`
- Status: `runnable`
- Difficulty: `intermediate`
- Category: State estimation
- Robot/fixture: `differential_drive_cube_v1`

## Objective

Submitted `RobotPolicy.step(robot)` code must fuse cumulative wheel encoder ticks, gyro yaw-rate samples, and delayed noisy AprilTag field-pose measurements into a `Pose2d` estimate in the episode-start field frame. The policy must submit an estimate every step while the environment drives a seeded trajectory past the tag field.

## Composition Decision

- Reuse: differential-drive robot primitive, odometry estimate scoring, MuJoCo world builder, render frame response, Rerun export, policy loader, and API route shape.
- Extend: lower-level odometry API with `DifferentialDriveStateEstimator.april_tag_measurements()` and replay samples for AprilTag pose updates.
- Reason: the embodiment, dynamics, and estimate-submission contract match the existing odometry challenge; the new surface is sensor-fusion input and a more realistic stochastic vision model.

## Embodiment And Environment

The public course uses the same 12 inch-class differential-drive cube in a bounded planar arena. The robot starts at `x=0`, `y=0`, `yaw=0`. Fixed AprilTags are placed around the course at known hidden field positions. The runner drives a deterministic seeded wheel-velocity profile, uses kinematic differential-drive integration with seeded left/right slip, and uses MuJoCo for replay rendering.

## Interface Contract

- Injected class: `DifferentialDriveStateEstimator`
- Base class: `MobileRobot`
- Public constants: `wheel_base_m`, `wheel_radius_m`, `ticks_per_rev`, `dt`, `max_steps`, `seed`
- Sensors: `get_encoder_values() -> tuple[int, int]`, `gyro() -> float`, `april_tag_measurements() -> tuple[AprilTagPoseEstimate, ...]`
- AprilTag measurement fields: `tag_id`, `timestamp`, `pose`, `distance_m`, `bearing_rad`, `position_std_m`, `yaw_std_rad`, `ambiguity`
- Outputs: `submit_odometry(Pose2d)`
- Hidden implementation: ground-truth pose, seeded trajectory, tag layout, slip scale, gyro bias/noise, tag visibility, dropout, latency, render details, and scoring.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Encoder values are cumulative integer ticks quantized from the environment's internal wheel commands. Gyro samples include seeded constant bias plus Gaussian yaw-rate noise. A seeded wheel-slip scale changes actual ground motion relative to encoder-derived wheel travel.

AprilTag measurements arrive at 10 Hz with about 0.12 s latency. Detection requires the tag to be inside the front camera field of view, within max range, and facing the camera. Dropout probability increases with range, edge-of-FOV bearing, and oblique viewing angle. Pose noise is Gaussian; position standard deviation grows with approximately squared distance and viewing angle, while yaw standard deviation is larger and similarly distance-dependent. Each measurement reports its own covariance proxy and ambiguity.

## Scenario Generation

The start pose is fixed in the field frame. Seed controls the internal trajectory profile, slip scale, gyro bias, gyro noise, tag dropout, and per-measurement pose noise. The episode runs for 16 seconds at 50 ms timesteps unless the robot leaves arena bounds.

## Success And Scoring

- Success: final position error <= 0.09 m, mean position error <= 0.07 m, final yaw error <= 4 degrees, and no collisions.
- Timeout: max step count reached without meeting the estimation tolerances.
- Collision/safety penalties: leaving arena bounds fails the run and reduces safety score.
- Score components: final position, mean trajectory position, yaw, and safety.
- Metrics returned: `final_position_error_m`, `mean_position_error_m`, `max_position_error_m`, `final_yaw_error_rad`, `mean_yaw_error_rad`, `excitation_distance_m`, `excitation_yaw_rad`.

## MuJoCo And Rerun Output

- MuJoCo cameras: `overview` and `front_camera`
- Render cadence: 10 Hz when MuJoCo is available
- Rerun static scene streams: arena bounds, route, tag layout, actual trajectory, submitted estimate trajectory
- Rerun time-series streams: robot pose, wheel commands, encoder ticks, gyro yaw rate, AprilTag pose measurements, odometry estimate, position error, yaw error, and MuJoCo images
- Replay metadata: timestep, render backend, robot dimensions, encoder resolution, noise settings, slip scales, tag layout, sample counts, and artifact status

## Demo Policy

The frontend starter reads encoder, gyro, and AprilTag measurement APIs and submits a placeholder `Pose2d(0, 0, 0)`. It demonstrates the required `april_tag_measurements` and `submit_odometry` calls without providing a sensor-fusion solution.

## Validation Plan

- Backend tests: catalog detail, API run with sample fusion policy, runner success, replay metadata, and AprilTag sample presence
- Frontend build: generated stub and starter policy compile in the app
- Browser run: open `/problems/diff_drive_state_estimation`, run the starter, confirm backend result
- MuJoCo verification: render frames appear in the replay response
- Rerun verification: `.rrd` artifact is attached and includes AprilTag sensor streams
