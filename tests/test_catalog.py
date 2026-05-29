from __future__ import annotations

import pytest

from roborank_envs.catalog import get_challenge, get_challenge_spec, list_challenges
from roborank_envs.runner import run_policy, run_policy_file


SAMPLE_POLICIES = {
    "diff_drive_reach_target": "pure_pursuit.py",
    "dock_to_charger": "dock_to_charger.py",
    "warehouse_aisle_avoidance": "warehouse_route_follower.py",
    "diff_drive_odometry": "odometry_baseline.py",
    "diff_drive_state_estimation": "state_estimation_fusion.py",
    "diff_drive_2d_slam": "slam_loop_closure.py",
    "diff_drive_lidar_maze": "lidar_maze_follower.py",
    "imu_collision_detection": "imu_collision_demo.py",
    "motor_torque_scale_control": "press_scale_pid.py",
    "quadrotor_gate_sequence": "quadrotor_gate_tracker.py",
    "cart_pole": "cart_pole_stabilizer.py",
    "cart_pole_minimum_phase": "cart_pole_minimum_phase.py",
    "trapezoidal_motion_profile": "trapezoidal_motion_profile.py",
    "inverse_kinematics_1": "inverse_kinematics_1.py",
    "inverse_kinematics_2": "inverse_kinematics_2.py",
    "inverse_kinematics_3": "inverse_kinematics_3.py",
}


def test_catalog_contains_current_roborank_challenges() -> None:
    challenges = list_challenges()

    assert len(challenges) == 16
    assert challenges[0]["id"] == "diff_drive_reach_target"
    assert {challenge["id"] for challenge in challenges} == {
        "diff_drive_reach_target",
        "diff_drive_odometry",
        "diff_drive_state_estimation",
        "diff_drive_2d_slam",
        "diff_drive_lidar_maze",
        "dock_to_charger",
        "warehouse_aisle_avoidance",
        "imu_collision_detection",
        "motor_torque_scale_control",
        "quadrotor_gate_sequence",
        "cart_pole",
        "cart_pole_minimum_phase",
        "trapezoidal_motion_profile",
        "inverse_kinematics_1",
        "inverse_kinematics_2",
        "inverse_kinematics_3",
    }


def test_get_challenge_returns_defensive_copy() -> None:
    challenge = get_challenge("diff_drive_reach_target")
    assert challenge is not None

    challenge["title"] = "mutated"

    fresh = get_challenge("diff_drive_reach_target")
    assert fresh is not None
    assert fresh["title"] == "Differential Drive: Reach Target"


def test_gtsam_is_available_for_environment_policies() -> None:
    import gtsam

    assert gtsam.Pose2()


@pytest.mark.parametrize(("challenge_id", "policy_path"), SAMPLE_POLICIES.items())
def test_sample_policy_runs_locally(challenge_id: str, policy_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBORANK_DISABLE_RERUN_EXPORT", "1")

    result = run_policy_file(
        challenge_id=challenge_id,
        policy_path=policy_path,
    )

    assert result.challenge_id == challenge_id
    assert result.metrics.success is True
    assert result.metrics.score > 80


def test_motion_profile_penalizes_acceleration_limit_violations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBORANK_DISABLE_RERUN_EXPORT", "1")

    class BadPolicy:
        def step(self, cart):
            cart.set_acceleration(cart.max_acceleration_mps2 * 2.0)

    challenge = get_challenge_spec("trapezoidal_motion_profile")
    assert challenge is not None

    result = run_policy(challenge=challenge, policy=BadPolicy(), seed=19, max_steps=4)

    assert result.metrics.success is False
    assert result.metrics.status == "limit_violation"
    assert result.metrics.acceleration_limit_violation_count == 4
    assert any(sample.acceleration_limit_violation for sample in result.replay.motion_profile_states)
