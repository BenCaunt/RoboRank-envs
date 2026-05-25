# Inverse Kinematics 1: Planar Arm

## Identity

- `challenge_id`: `inverse_kinematics_1`
- Status: `runnable`
- Difficulty: Beginner
- Category: Kinematics
- Robot/fixture: `planar_2link_arm_v1`

## Objective

Write a `RobotPolicy` that computes shoulder and elbow angles for a two-link planar arm. Each policy step receives one target point and must call `arm.submit_joint_angles([shoulder, elbow])`.

## Composition Decision

- Reuse: catalog models, sandboxed policy execution, run response, MuJoCo render frames, Rerun artifact export, frontend challenge composer.
- New infrastructure: shared manipulator spec, inverse-kinematics API, analytic IK runner, and manipulator MuJoCo visualization world.
- Reason: the task is a static kinematic solve, not a differential-drive or dynamic-control problem.

## Embodiment And Environment

The shoulder joint is fixed at the origin of the horizontal `XY` plane. Link lengths are `L1 = 0.42 m` and `L2 = 0.34 m`. The target is reachable by construction and is exposed as `(x, y, z=0)`.

## Interface Contract

- Injected class: `InverseKinematicsTask`
- Base class: `KinematicMechanism`
- Public constants: `mechanism`, `joint_count`, `link_1_m`, `link_2_m`, `joint_limits_rad`, `tolerance_m`, `dt`, `target_index`, `target_count`
- Sensors: `get_target() -> IKTarget`, `joint_limits() -> tuple[tuple[float, float], ...]`
- Actuators: `submit_joint_angles(angles) -> None`
- Hidden implementation: target generation, forward-kinematics scoring, rendering, and Rerun export.

## Stochastic Contract

Public runs are deterministic for a fixed seed. Targets are sampled from reachable joint configurations and then exposed only as target coordinates.

## Success And Scoring

- Max end-effector error: `<= 0.015 m`
- Mean end-effector error: `<= 0.008 m`
- Joint limit violations: `0`
- Score components: mean accuracy, worst-case accuracy, and joint-limit compliance.

## Replay Contract

MuJoCo renders the arm, target points, and current target. Rerun logs the target path, end-effector path, per-target error, joint angles, and encoded MuJoCo frames.

## Demo Policy

The frontend starter reads the target and joint limits, then submits zero angles. It exercises the API and replay path without solving the kinematics.

## Validation Plan

Run backend tests for catalog, runner success with the sample analytic policy, bad submission handling, render frames, and Rerun artifact creation. Run the frontend build and browser check for `/problems/inverse_kinematics_1`.
