# IMU Collision Detection

## Identity

- `challenge_id`: `imu_collision_detection`
- Status: `runnable`
- Difficulty: `intermediate`
- Category: Signal processing
- Robot/fixture: 12 inch-class differential-drive mobile probe

## Objective

Submitted `RobotPolicy` code receives noisy IMU samples and must submit whether the current sample indicates physical contact. The frontend demo only shows how to read `imu()` and call `submit_collision_decision`.

## Composition Decision

- Reuse: differential-drive robot primitive, MuJoCo world builder, render frame response, Rerun artifact writer, policy loader, and API run endpoint.
- Extend: policy API with `CollisionProbe`, replay traces with IMU samples and classifier decisions, and runner routing with `runner="imu_collision"`.
- New infrastructure: a small `ImuCollisionRunner` that drives a hidden course and scores classifier latency and false positives.
- Reason: the task is a signal-classification challenge, so user code should not command wheels or read target pose.

## Embodiment And Environment

The hidden runner places the mobile probe in a bounded MuJoCo arena with one seeded circular post. The probe drives forward under hidden wheel commands until first contact, then records a short post-collision window. Public fallback remains kinematic if MuJoCo cannot import.

## Interface Contract

- Injected class: `CollisionProbe`
- Base class: `MobileRobot`
- Public constants: `dt`, `max_steps`, `seed`
- Sensors: `imu() -> ImuSample`
- Actuators: `submit_collision_decision(contact: bool, severity: str) -> None`
- Hidden implementation: drive commands, contact labels, MuJoCo internals, terrain vibration, and impact generation.

## Stochastic Contract

Runs are deterministic for a fixed seed. Start pose, post pose, IMU Gaussian noise, vibration phase, and mild terrain bump timing are seed dependent. Contact impulses saturate the public signal enough for a dummy threshold baseline.

## Scenario Generation

The seeded course starts the probe near the left side of the arena, places one collision post on the forward path, and runs at `dt=0.05` for up to 4 seconds with a 0.5 second post-contact trace.

## Success And Scoring

- Success: first collision is detected within 80 ms.
- Timeout: no accepted detection before the latency window ends.
- Collision/safety penalties: this challenge expects one contact; false positives before contact are penalized.
- Score components: 60 detection, 20 latency, 20 precision.
- Metrics returned: standard score/status fields plus replay metadata for detection latency, false positives, IMU sample count, and decision count.

## MuJoCo And Rerun Output

- MuJoCo cameras: `overview` and `front_camera`
- Render cadence: 0.1 seconds plus the contact frame
- Rerun static scene streams: arena bounds, target marker, obstacle, robot box, trajectory, and summary
- Rerun time-series streams: robot transforms, hidden controls, IMU axes, classifier contact/severity, and encoded MuJoCo frames
- Replay metadata: backend, simulation mode, render status, seed, latency, false positives, and artifact count

## Demo Policy

The frontend starter reads `robot.imu()`, touches the acceleration and gyro fields, and submits `False, "none"` on every step. It demonstrates the required output API without providing the thresholding logic needed to solve the classifier.

## Validation Plan

- Backend tests: catalog detail, sample policy run, trace fields, render frames, and Rerun artifact
- Frontend build: generated stub and starter code compile
- Browser run: open `/problems/imu_collision_detection`, run the starter policy, inspect MuJoCo and Rerun panes
- MuJoCo verification: render frames are present when the dependency is available
- Rerun verification: `.rrd` artifact is returned unless optional dependency import/export fails
