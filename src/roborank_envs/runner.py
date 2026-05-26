from __future__ import annotations

from pathlib import Path

from roborank_envs.catalog import get_challenge_spec
from roborank_envs.models import ChallengeSpec, RunResult
from roborank_envs.policy_loader import load_policy
from roborank_envs.simulation.cart_pole import CartPoleRunner
from roborank_envs.simulation.diff_drive import (
    DifferentialDriveOdometryRunner,
    DifferentialDriveRunner,
    DifferentialDriveSlamRunner,
)
from roborank_envs.simulation.imu_collision import ImuCollisionRunner
from roborank_envs.simulation.inverse_kinematics import InverseKinematicsRunner
from roborank_envs.simulation.motor_stand import MotorStandRunner
from roborank_envs.simulation.quadrotor import QuadrotorGateRunner


class EnvironmentRunError(RuntimeError):
    pass


def run_policy_file(
    *,
    challenge_id: str,
    policy_path: str | Path,
    seed: int | None = None,
    max_steps: int | None = None,
) -> RunResult:
    challenge = get_challenge_spec(challenge_id)
    if challenge is None:
        raise EnvironmentRunError(f"Unknown challenge id: {challenge_id}")

    policy = load_policy(policy_path=str(policy_path))
    return run_policy(challenge=challenge, policy=policy, seed=seed, max_steps=max_steps)


def run_policy(
    *,
    challenge: ChallengeSpec,
    policy: object,
    seed: int | None = None,
    max_steps: int | None = None,
) -> RunResult:
    runner_name = str(challenge.defaults.get("runner", "diff_drive"))
    runner = _runner_for_challenge(challenge, runner_name=runner_name)
    return runner.run(policy, seed=_evaluation_seed(challenge) if seed is None else int(seed), max_steps=max_steps)


def _runner_for_challenge(challenge: ChallengeSpec, *, runner_name: str) -> object:
    if challenge.robot.type == "motor_stand":
        return MotorStandRunner(challenge)
    if runner_name == "cart_pole":
        return CartPoleRunner(challenge)
    if runner_name == "quadrotor_gate_sequence":
        return QuadrotorGateRunner(challenge)
    if runner_name == "imu_collision":
        return ImuCollisionRunner(challenge)
    if runner_name == "inverse_kinematics":
        return InverseKinematicsRunner(challenge)
    if runner_name in {"diff_drive_odometry", "diff_drive_state_estimation"}:
        return DifferentialDriveOdometryRunner(challenge)
    if runner_name == "diff_drive_slam":
        return DifferentialDriveSlamRunner(challenge)
    if challenge.robot.type == "differential_drive":
        return DifferentialDriveRunner(challenge)
    raise EnvironmentRunError(f"No runner available for challenge {challenge.id!r}.")


def _evaluation_seed(challenge: ChallengeSpec) -> int:
    try:
        return int(challenge.defaults.get("evaluation_seed", 0))
    except (TypeError, ValueError):
        return 0
