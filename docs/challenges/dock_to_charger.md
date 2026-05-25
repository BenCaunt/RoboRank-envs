# Dock to Charging Pad

## Identity

- challenge_id: `dock_to_charger`
- title: Dock to Charging Pad
- difficulty: intermediate
- status: runnable
- category: Closed-loop autonomy

## User-Facing Objective

Command a 12 inch-class differential-drive robot into a charging bay. The policy must use a noisy front-camera marker pose estimate, wheel encoder ticks, and gyro yaw rate to finish within 0.05 m of the pad center and within 3 degrees of the dock heading without contact.

## Composition Decision

This reuses and extends the existing differential-drive runner because the embodiment, action space, MuJoCo stepping, render pipeline, and Rerun export already match a planar docking task. The extension adds a docking scenario, camera-style charger pose observations, encoder/gyro API fields, docking success logic, and docking-specific scoring.

## Embodiment And Environment

The robot is `differential_drive_cube_v1` in a bounded indoor bay. The charging pad is a non-contact visual target with two bumper obstacles near the final pose plus seeded clutter outside the nominal approach lane. The robot starts in front of the bay with randomized lateral offset and heading error.

## Interface Contract

The platform instantiates `RobotPolicy()` and calls `step(robot)` with a hidden `DifferentialDrive` object. Public methods and constants:

- `wheel_base_m`, `max_wheel_velocity_mps`, `ticks_per_rev`
- `camera() -> np.ndarray`
- `get_encoder_values() -> tuple[float, float]`
- `gyro() -> float`
- `charger_pose() -> Pose2d | None`
- `set_wheel_velocity(left_mps, right_mps) -> None`

`charger_pose()` is the charging pad pose in the robot frame: `x` forward, `y` left, and `yaw` as the pad heading expressed relative to the robot heading. MuJoCo internals, exact target pose, collision checks, dropout timing, and scoring stay hidden.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Start pose, dock pose, obstacle offsets, camera-pose bias, and one short dropout window vary by seed. The charger estimate includes Gaussian position/yaw noise and can return `None` when the marker is outside the camera field of view or during dropout.

## Scenario Generation

The target is placed near the right side of the bay with small seeded pose variation. The robot starts from the left with seeded lateral and yaw offsets. Bumper obstacles frame the dock, and additional clutter sits away from the nominal approach route.

## Success And Scoring

Pass criteria are:

- final position error <= 0.05 m
- final yaw error <= 3 degrees
- collision_count = 0
- timeout <= 25 seconds

Scoring allocates points for docking success, yaw alignment, time remaining, safety, smoothness, and wheel effort.

## Replay Contract

The run returns MuJoCo overview and front-camera PNG frames. Rerun logs the arena, charger target heading, robot trajectory, wheel controls, lidar samples, charger pose visibility and estimates, and MuJoCo images. The run response includes a `.rrd` artifact when Rerun is installed.

## Demo Policy

The frontend starter calls the camera, encoder, gyro, and `charger_pose()` APIs, rotates slowly when the marker is not visible, and creeps forward only while the marker is still distant. It intentionally stops short of the docking tolerance and is not a visual-servo solution.

## Validation Plan

- Backend catalog detail and run endpoint tests for `dock_to_charger`.
- Runner success/failure shape, render frame presence, charger pose trace, and Rerun artifact checks.
- Frontend build verifies generated stub and challenge data.
- Browser verification opens `/problems/dock_to_charger`, runs the starter policy, checks backend execution, MuJoCo frames, and Rerun availability.
