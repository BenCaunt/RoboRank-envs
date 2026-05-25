# Inverse Kinematics 2: Turret Arm

## Identity

- `challenge_id`: `inverse_kinematics_2`
- Status: `runnable`
- Difficulty: Intermediate
- Category: Kinematics
- Robot/fixture: `turret_2link_arm_v1`

## Objective

Write a `RobotPolicy` that computes turret yaw, shoulder pitch, and elbow pitch for a two-link arm reaching 3D target points. Each policy step must call `arm.submit_joint_angles([yaw, shoulder, elbow])`.

## Composition Decision

- Reuse: shared manipulator challenge infrastructure, sandbox execution, run response, MuJoCo rendering, Rerun export, and frontend composer.
- New infrastructure: challenge-specific catalog metadata, frontend diagram, starter, docs, and sample policy.
- Reason: this is the 3D extension of the planar IK runner and can share the same analytic scorer.

## Embodiment And Environment

The turret yaw joint rotates the radial working plane about world `+z`. The shoulder and elbow then solve a two-link reach in radial distance and height. Link lengths are `L1 = 0.46 m`, `L2 = 0.38 m`, and the shoulder pivot is `0.18 m` above the ground plane.

## Interface Contract

- Injected class: `InverseKinematicsTask`
- Base class: `KinematicMechanism`
- Public constants: `mechanism`, `joint_count`, `link_1_m`, `link_2_m`, `base_height_m`, `joint_limits_rad`, `tolerance_m`, `dt`, `target_index`, `target_count`
- Sensors: `get_target() -> IKTarget`, `joint_limits() -> tuple[tuple[float, float], ...]`
- Actuators: `submit_joint_angles(angles) -> None`
- Hidden implementation: target generation, 3D forward-kinematics scoring, rendering, and Rerun export.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Targets are sampled from reachable yaw, shoulder, and elbow configurations and exposed as `(x, y, z)` coordinates.

## Success And Scoring

- Max wrist error: `<= 0.022 m`
- Mean wrist error: `<= 0.012 m`
- Joint limit violations: `0`
- Score components: mean accuracy, worst-case accuracy, and joint-limit compliance.

## Replay Contract

MuJoCo renders overview and side cameras of the turreted arm. Rerun logs the target sequence, end-effector path, per-target error, joint angles, and encoded MuJoCo frames.

## Demo Policy

The frontend starter reads the target and joint limits, then submits zero angles. It exercises the API and replay path without solving the 3D kinematics.

## Validation Plan

Run backend tests for catalog, runner success with the sample analytic policy, bad submission handling, render frames, and Rerun artifact creation. Run the frontend build and browser check for `/problems/inverse_kinematics_2`.
