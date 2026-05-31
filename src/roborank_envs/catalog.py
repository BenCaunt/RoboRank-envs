from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from roborank_envs.models import ChallengeSpec


CatalogEntry = dict[str, Any]


class CatalogError(RuntimeError):
    """Raised when packaged challenge data is malformed."""


def list_challenges() -> list[CatalogEntry]:
    return [copy.deepcopy(challenge) for challenge in _load_catalog()]


def get_challenge(challenge_id: str) -> CatalogEntry | None:
    for challenge in _load_catalog():
        if challenge["id"] == challenge_id:
            return copy.deepcopy(challenge)
    return None


def list_challenge_specs() -> list[ChallengeSpec]:
    return [ChallengeSpec.model_validate(challenge) for challenge in _load_catalog()]


def get_challenge_spec(challenge_id: str) -> ChallengeSpec | None:
    challenge = get_challenge(challenge_id)
    if challenge is None:
        return None
    return ChallengeSpec.model_validate(challenge)


def validate_catalog() -> list[CatalogEntry]:
    return list_challenges()


@lru_cache(maxsize=1)
def _load_catalog() -> tuple[CatalogEntry, ...]:
    resource = files("roborank_envs").joinpath("catalog_data.json")
    with resource.open("r", encoding="utf-8") as catalog_file:
        raw_catalog = json.load(catalog_file)

    if not isinstance(raw_catalog, list):
        raise CatalogError("catalog_data.json must contain a list of challenges.")

    ids: set[str] = set()
    catalog: list[CatalogEntry] = []
    required_keys = {
        "id",
        "title",
        "difficulty",
        "description",
        "robot",
        "sensors",
        "actuators",
        "objective",
        "success_conditions",
        "scoring",
        "defaults",
    }

    for index, challenge in enumerate(raw_catalog):
        if not isinstance(challenge, dict):
            raise CatalogError(f"Catalog entry {index} must be an object.")
        missing = required_keys - set(challenge)
        if missing:
            challenge_id = challenge.get("id", f"entry {index}")
            raise CatalogError(f"Challenge {challenge_id!r} is missing required keys: {sorted(missing)}.")
        challenge_id = challenge["id"]
        if not isinstance(challenge_id, str) or not challenge_id:
            raise CatalogError(f"Catalog entry {index} has an invalid id.")
        if challenge_id in ids:
            raise CatalogError(f"Duplicate challenge id {challenge_id!r}.")
        ids.add(challenge_id)
        catalog.append(challenge)

    return tuple(catalog)


DIFF_DRIVE_REACH_TARGET = get_challenge_spec("diff_drive_reach_target")
DIFF_DRIVE_ODOMETRY = get_challenge_spec("diff_drive_odometry")
DIFF_DRIVE_STATE_ESTIMATION = get_challenge_spec("diff_drive_state_estimation")
DIFF_DRIVE_2D_SLAM = get_challenge_spec("diff_drive_2d_slam")
DIFF_DRIVE_LIDAR_MAZE = get_challenge_spec("diff_drive_lidar_maze")
DOCK_TO_CHARGER = get_challenge_spec("dock_to_charger")
WAREHOUSE_AISLE_AVOIDANCE = get_challenge_spec("warehouse_aisle_avoidance")
IMU_COLLISION_DETECTION = get_challenge_spec("imu_collision_detection")
MOTOR_TORQUE_SCALE_CONTROL = get_challenge_spec("motor_torque_scale_control")
QUADROTOR_GATE_SEQUENCE = get_challenge_spec("quadrotor_gate_sequence")
CART_POLE = get_challenge_spec("cart_pole")
CART_POLE_MINIMUM_PHASE = get_challenge_spec("cart_pole_minimum_phase")
TRAPEZOIDAL_MOTION_PROFILE = get_challenge_spec("trapezoidal_motion_profile")
KALMAN_ACCELERATION_ESTIMATION = get_challenge_spec("kalman_acceleration_estimation")
INVERSE_KINEMATICS_1 = get_challenge_spec("inverse_kinematics_1")
INVERSE_KINEMATICS_2 = get_challenge_spec("inverse_kinematics_2")
INVERSE_KINEMATICS_3 = get_challenge_spec("inverse_kinematics_3")
