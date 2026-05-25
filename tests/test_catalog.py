from __future__ import annotations

from roborank_envs.catalog import get_challenge, list_challenges


def test_catalog_contains_current_roborank_challenges() -> None:
    challenges = list_challenges()

    assert len(challenges) == 14
    assert challenges[0]["id"] == "diff_drive_reach_target"
    assert {challenge["id"] for challenge in challenges} == {
        "diff_drive_reach_target",
        "diff_drive_odometry",
        "diff_drive_state_estimation",
        "diff_drive_2d_slam",
        "dock_to_charger",
        "warehouse_aisle_avoidance",
        "imu_collision_detection",
        "motor_torque_scale_control",
        "quadrotor_gate_sequence",
        "cart_pole",
        "cart_pole_minimum_phase",
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
