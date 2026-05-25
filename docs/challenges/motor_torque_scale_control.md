# Motor Torque: Press Scale

## Identity

- `challenge_id`: `motor_torque_scale_control`
- Status: `runnable`
- Difficulty: Intermediate
- Category: Actuation and control
- Robot/fixture: `motor_stand_scale_v1`

## Objective

Write `RobotPolicy.step(motor)` to command current in a current-controlled motor so the shaft tip presses a digital scale to the seeded target force quickly, with bounded overshoot.

## Composition Decision

- Reuse: Existing FastAPI run endpoint, policy loader, challenge composer, MuJoCo render response shape, and Rerun artifact plumbing.
- Extend: Shared catalog/model schema and frontend replay normalization now include force-control traces.
- New infrastructure: Dedicated motor stand runner and MuJoCo fixture world.
- Reason: A motor pressing a scale is a contact/actuator fixture, not planar differential-drive motion, so it needs different dynamics, sensors, controls, scoring, and render geometry.

## Embodiment And Environment

The fixture is a bench-top motor stand with a 0.18 m shaft pressing a small scale plate. The simulation models current lag, torque constant variation, shaft-tip force, scale compliance, scale response lag, sensor noise, and small force losses.

## Interface Contract

- Injected class: `CurrentControlledMotor`
- Base class: `Actuator`
- Public constants: `shaft_length_m`, `kt_nm_per_amp`, `max_current_a`, `target_force_n`, `dt`
- Sensors: `target_force()`, `current()`, `scale_force()`
- Actuators: `set_current(amps)`
- Hidden implementation: Electrical lag, compliance, scale dynamics, noise, seeded physical variation, rendering, and scoring.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Target force is sampled from 0.82 N to 1.08 N. Current measurements include small bias and Gaussian noise. Scale readings include small bias, Gaussian noise, and compliance hysteresis. Motor constant, current lag, scale lag, and contact loss vary by seed.

## Scenario Generation

Each seed creates one target force and one fixture calibration. The episode starts at zero current and zero force. The default timestep is 0.02 s and the timeout is 5.0 s.

## Success And Scoring

- Success: Mean final scale-force error is at most 0.2 N, overshoot is at most 8%, and settling time is at most 1.5 s.
- Timeout: 5.0 s.
- Collision/safety penalties: Not applicable; overshoot is the safety constraint.
- Score components: Force accuracy, settling time, overshoot, current smoothness, and current effort.
- Metrics returned: Target force, final force error, mean absolute force error, settling time, overshoot, peak force, current effort, and current jerk.

## MuJoCo And Rerun Output

- MuJoCo cameras: `overview`, `scale_closeup`
- Render cadence: 0.1 s
- Rerun static scene streams: Fixture base, scale plate, summary text.
- Rerun time-series streams: Shaft line, scale force, measured current, force error, shaft angle, torque, target force, current command, and MuJoCo frames.
- Replay metadata: Seed, timestep, fixture dimensions, target force, backend mode, render cameras, render count, render errors, and Rerun export errors.

## Demo Policy

The frontend starter reads current and scale-force sensors, computes an approximate feedforward current from public constants, and commands a bounded sinusoidal current. It visibly moves the shaft and scale response without using feedback control or serving as a tuned solution.

## Validation Plan

- Backend tests: Catalog detail, sample PID run, bad action validation, replay metadata, render frames, and Rerun artifact presence.
- Frontend build: TypeScript and Vite production build.
- Browser run: Open `/problems/motor_torque_scale_control`, verify stub/starter, run the demo, and inspect MuJoCo/Rerun panes.
- MuJoCo verification: Confirm render frames from both fixture cameras are returned.
- Rerun verification: Confirm a `.rrd` artifact is produced for the run.
