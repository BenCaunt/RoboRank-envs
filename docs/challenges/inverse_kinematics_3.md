# Inverse Kinematics 3: SCARA Drawing Arm

## Identity

- `challenge_id`: `inverse_kinematics_3`
- Status: `runnable`
- Difficulty: Advanced
- Category: Kinematics
- Robot/fixture: `five_bar_scara_v1`

## Objective

Write a `RobotPolicy` that computes the two motor angles for a five-bar SCARA-style drawing mechanism. Each policy step receives the next pen target and must call `arm.submit_joint_angles([left_motor, right_motor])`.

## Composition Decision

- Reuse: shared inverse-kinematics runner, manipulator spec, sandbox execution, MuJoCo visual world, Rerun export, and frontend composer.
- New infrastructure: five-bar catalog metadata, drawing trajectory generator, frontend diagram, starter, docs, and sample policy.
- Reason: the drawing arm is still an analytic IK task, but the closed-chain forward kinematics and trajectory scoring make it the advanced variant.

## Embodiment And Environment

Two base motors are separated by `0.36 m`. Each motor drives a `0.30 m` proximal link, and each elbow connects to the pen with a `0.34 m` distal link. The pen tip is the upper intersection of the two distal-link circles in the horizontal drawing plane.

## Interface Contract

- Injected class: `InverseKinematicsTask`
- Base class: `KinematicMechanism`
- Public constants: `mechanism`, `joint_count`, `link_1_m`, `link_2_m`, `base_spacing_m`, `joint_limits_rad`, `tolerance_m`, `dt`, `target_index`, `target_count`
- Sensors: `get_target() -> IKTarget`, `joint_limits() -> tuple[tuple[float, float], ...]`, `target_index`, `target_count`
- Actuators: `submit_joint_angles(angles) -> None`
- Hidden implementation: drawing stroke generation, closed-chain forward kinematics, rendering, and Rerun export.

## Stochastic Contract

Public runs are deterministic for a fixed seed. The target stroke is generated inside the reachable workspace. Hidden variants may alter the drawing stroke while keeping the same declared mechanism dimensions.

## Success And Scoring

- Max pen-tip error: `<= 0.020 m`
- Mean pen-tip error: `<= 0.010 m`
- Joint limit violations: `0`
- Score components: mean tracking accuracy, worst-case tracking accuracy, and joint-limit compliance.

## Replay Contract

MuJoCo renders overview and top cameras of the five-bar arm, the target stroke, and current pen target. Rerun logs the target stroke, pen path, per-target error, motor angles, and encoded MuJoCo frames.

## Demo Policy

The frontend starter reads the target and joint limits, then submits zero angles. It exercises the API and replay path without revealing the closed-chain IK solution.

## Validation Plan

Run backend tests for catalog, runner success with the sample analytic policy, bad submission handling, render frames, and Rerun artifact creation. Run the frontend build and browser check for `/problems/inverse_kinematics_3`.
