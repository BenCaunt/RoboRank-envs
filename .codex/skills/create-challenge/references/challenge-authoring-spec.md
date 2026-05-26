# Challenge Authoring Spec

Use this reference when adding robotics or computer-vision coding challenges to the `leetcode-but-robotics` app.

## Current Integration Map

- Frontend challenge model: `frontend/src/types.ts`
- Frontend challenge data, interface members, stub generation, starter code: `frontend/src/data.ts`
- Problem rendering, run controls, MuJoCo and Rerun panes: `frontend/src/App.tsx`
- Backend API models: `backend/app/models.py`
- Backend catalog: `backend/app/catalog.py`
- Injected user policy API: `backend/app/policy_api.py`
- Policy execution/injected globals: `backend/app/policy_loader.py`
- API runner routing: `backend/app/main.py`
- Existing differential-drive runner: `backend/app/simulation/diff_drive.py`
- Existing MuJoCo world builder: `backend/app/simulation/mujoco_world.py`
- Existing Rerun export: `backend/app/simulation/rerun_export.py`
- Computer-vision runner patterns: `backend/app/simulation/computer_vision.py`
- Shared robot primitives: `backend/app/simulation/robot_primitives.py`
- Backend samples and tests: `backend/samples/policies/`, `backend/tests/`

## Challenge Brief Schema

Create a brief for substantial challenges, especially when adding a new embodiment, stochastic sensor model, or harness. Store it in a repo-local docs or specs folder if one exists; otherwise create a minimal `docs/challenges/<challenge-id>.md`.

Required sections:

- Identity: `challenge_id`, title, difficulty, status (`runnable` or `draft`), category.
- User-facing objective: what the submitted policy must accomplish.
- Composition decision: existing harness reused, extended, or new; include why.
- Embodiment and environment: robot, fixture, arena/course, relevant physical assumptions.
- Interface contract: injected class name, base class, exposed constants, sensors, actuators, and hidden implementation.
- Stochastic contract: seeded randomness, sensor noise, latency, dropout, bias, physical variation, and which runs are deterministic.
- Scenario generation: seed inputs, start states, target/course randomization, obstacles or disturbances.
- Success and scoring: pass/fail criteria, score components, timeouts, penalties, and metric names.
- Replay contract: MuJoCo render cameras/frames when physics-backed, Rerun streams, sensor/image traces, controls or submitted outputs, and metadata.
- Demo policy: behavior that visibly exercises the simulation and critical API calls without solving the task or relying on hidden internals.
- Validation plan: tests, frontend build, browser run, MuJoCo/Rerun checks.

## Composition Rules

Prefer reuse in this order:

1. Catalog/frontend-only spec update, only when the user explicitly wants a draft.
2. Existing differential-drive runner for planar target/obstacle tasks using wheel velocity commands, solved pose, target, obstacles, and lidar.
3. Extension of the differential-drive API/runner for new low-level sensors, visual servoing, odometry, docking, route following, or alternative scoring that still uses planar differential-drive dynamics.
4. Existing computer-vision runner for image-only perception tasks with seeded images, OpenCV policy access, submitted detections/classifications/estimates, and Rerun image replay.
5. New runner plus MuJoCo world builder for different physics: quadrotor gates, motor torque fixtures, manipulators, contact/signal tasks, or anything that cannot honestly be represented by the existing planar base.
6. Shared primitives when two or more challenges will use the same robot, fixture, image dataset, sensor model, scorer, or replay exporter.

Do not create one-off copies of a runner when parameterization or a small shared helper would make the relationship explicit.

## User Policy Contract

The submitted Python module should define:

```python
class RobotPolicy:
    def step(self, robot):
        ...
```

The platform instantiates `RobotPolicy()` and injects a runtime object into `step`. Avoid public `reset` requirements; re-instantiating the policy per run is the default. Actions or answers should be set through methods on the injected object, such as `robot.set_wheel_velocity(...)`, `motor.set_current(...)`, or `vision.submit_points(...)`.

Expose constants and sensor methods on the injected object. Add commented stubs in frontend starter code and problem text, for example:

```python
# Definition for the hidden mobile robot API.
# The simulator implements these objects and injects one DifferentialDrive instance into RobotPolicy.step.
#
# class Pose2d:
#     x: float
#     y: float
#     yaw: float
#
# class DifferentialDrive(MobileRobot):
#     wheel_base_m: float = 0.2667
#     max_wheel_velocity_mps: float = 1.25
#
#     def get_pose() -> Pose2d:
#         # Robot pose in the world frame.
#         # Public runner is exact; hidden variants must declare pose covariance here.
```

Any stochastic sensor behavior must be visible in the stub text or problem contract. Do not make users infer noise, delay, dropout, quantization, bias, hidden randomization, or actuator saturation from implementation code.

## Backend Requirements

For runnable challenges:

- `ChallengeSpec` should describe the public problem and defaults needed by the runner.
- Policy APIs should use dataclasses/simple runtime classes for user-facing values and commands.
- Policy loader globals should include every user-visible type used in starter code.
- Runners should seed `random` and `numpy`, create a scenario or image batch, call `policy.step(robot_or_task)`, consume the command/submission, score, and return `RunResult`.
- MuJoCo world builders should render at least one meaningful camera. Mobile robot challenges should usually include `overview`; camera/perception challenges should also include the relevant sensor camera.
- Computer-vision challenges should not fabricate MuJoCo frames. Mark replay metadata with `mujoco_backend="not_applicable"`, `simulation_mode="offline_image_dataset"` or a more specific mode, and zero render frames.
- Replay metadata should report MuJoCo backend, simulation mode, render camera names, render frame count, seed, timestep, and robot/fixture dimensions where useful.
- Rerun export should log static scene, robot/fixture state, controls, important sensor/image traces, submitted outputs, summary text, and encoded MuJoCo frames when they exist.
- Fallbacks may keep grading usable when MuJoCo/Rerun is not installed, but the response should expose the render/export error in metadata.
- `opencv-python` is a required dependency for computer-vision challenges, and submitted policies should be able to `import cv2`.

## Frontend Requirements

- Add reusable `InterfaceMember` and stub types when the API surface is shared by future challenges.
- Use generated stubs through the existing `composeChallenge` flow; avoid hand-written one-off code blocks.
- Keep `sensors` and `actuators` derived from the interface contract when possible.
- Mark draft/spec-only challenges clearly and do not route them to a nonexistent backend runner.
- Give computer-vision challenges replay capabilities with Rerun enabled and MuJoCo disabled so the UI does not promise physics frames.
- For runnable additions, update any playability/routing checks that currently assume only one runnable challenge.
- Starter code should demonstrate actuator/submission and replay behavior, remain short, avoid hidden implementation details, and avoid serving as a reference solution.

## Demo Policy Guidance

Demo code should be intentionally modest and should not solve the challenge. Put tuned baselines, reference controllers, route followers, estimators, and threshold classifiers in `backend/samples/policies/` or tests instead of the frontend starter.

- Differential drive: command low forward velocity with a sinusoidal angular component or a small sensor-reactive safety behavior such as stopping for close lidar.
- Odometry/sensor tasks: read exposed sensors, submit the required placeholder output such as `Pose2d(0, 0, 0)`, and command slow bounded movement so traces are populated.
- Motor torque scale: command a small sinusoidal or ramped current within limits.
- Drone gates: command conservative hover/forward motion with bounded rotor or velocity commands.
- Computer vision detection tasks: read `vision.image()`, optionally call one basic OpenCV transformation to prove `cv2` access, and submit an empty or deliberately incomplete answer such as `vision.submit_points([])`.
- Classifier tasks: call the required submission API with a neutral or deliberately trivial decision, such as `submit_collision_decision(False, "none")`, so the user sees the output path without receiving a passable classifier.

Do not add demo policies that depend on private scenario state, solve hidden variants by reading internals, or pass the public task through a compact baseline algorithm.

## Validation Checklist

Run the relevant subset:

```bash
cd backend && uv run pytest
npm --prefix frontend run build
```

For runnable frontend changes, use the in-app browser:

- Open `/problems/<challenge-id>`.
- Confirm the problem statement and editor show the generated API stub.
- Run the demo policy.
- Confirm the backend endpoint is used instead of a synthetic demo response.
- Confirm MuJoCo render frames are visible for physics-backed challenges, or that the MuJoCo view is disabled by design for computer-vision challenges.
- Confirm the Rerun panel has a `.rrd` artifact, or document the local dependency error.

The final response should state exactly what was implemented and what validation passed.
