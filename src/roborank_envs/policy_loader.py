from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from roborank_envs.policy_api import (
    Actuator,
    AerialRobot,
    AprilTagPoseEstimate,
    CartPole,
    CartPoleState,
    CircleObstacle,
    CollisionDecision,
    CollisionProbe,
    ControlSystem,
    CurrentControlledMotor,
    DifferentialDrive,
    DifferentialDriveOdometry,
    DifferentialDriveSlam,
    DifferentialDriveStateEstimator,
    Gate3d,
    IKTarget,
    ImuSample,
    InverseKinematicsTask,
    KinematicMechanism,
    MapPoint2d,
    MinimumPhaseState,
    MobileRobot,
    MotionState1D,
    MotionTarget1D,
    Pose2d,
    Pose3d,
    ProfiledCart1D,
    Quadrotor,
    RobotPolicyProtocol,
    Target2d,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
POLICY_GLOBALS = {
    "Actuator": Actuator,
    "AerialRobot": AerialRobot,
    "AprilTagPoseEstimate": AprilTagPoseEstimate,
    "CartPole": CartPole,
    "CartPoleState": CartPoleState,
    "CircleObstacle": CircleObstacle,
    "CollisionDecision": CollisionDecision,
    "CollisionProbe": CollisionProbe,
    "ControlSystem": ControlSystem,
    "CurrentControlledMotor": CurrentControlledMotor,
    "DifferentialDrive": DifferentialDrive,
    "DifferentialDriveOdometry": DifferentialDriveOdometry,
    "DifferentialDriveSlam": DifferentialDriveSlam,
    "DifferentialDriveStateEstimator": DifferentialDriveStateEstimator,
    "Gate3d": Gate3d,
    "IKTarget": IKTarget,
    "ImuSample": ImuSample,
    "InverseKinematicsTask": InverseKinematicsTask,
    "KinematicMechanism": KinematicMechanism,
    "MapPoint2d": MapPoint2d,
    "MinimumPhaseState": MinimumPhaseState,
    "MobileRobot": MobileRobot,
    "MotionState1D": MotionState1D,
    "MotionTarget1D": MotionTarget1D,
    "np": np,
    "numpy": np,
    "Pose2d": Pose2d,
    "Pose3d": Pose3d,
    "ProfiledCart1D": ProfiledCart1D,
    "Quadrotor": Quadrotor,
    "Target2d": Target2d,
}


class PolicyLoadError(ValueError):
    pass


def load_policy(
    *,
    policy_path: str | None = None,
    policy_module: str | None = None,
    policy_source: str | None = None,
) -> RobotPolicyProtocol:
    provided_count = sum(bool(value) for value in (policy_path, policy_module, policy_source))
    if provided_count != 1:
        raise PolicyLoadError("Provide exactly one of policy_path, policy_module, or policy_source.")

    if policy_path:
        module = _load_module_from_path(policy_path)
    elif policy_module:
        module = _load_module_by_name(policy_module)
    else:
        module = _load_module_from_source(policy_source)
    policy_class = getattr(module, "RobotPolicy", None)
    if policy_class is None:
        raise PolicyLoadError("Policy module must define a RobotPolicy class.")

    try:
        policy = policy_class()
    except TypeError as exc:
        raise PolicyLoadError("RobotPolicy must be constructible with no arguments.") from exc

    if not callable(getattr(policy, "step", None)):
        raise PolicyLoadError("RobotPolicy must define step(robot).")

    return policy


def _load_module_by_name(module_name: str | None) -> ModuleType:
    if not module_name:
        raise PolicyLoadError("Missing policy_module.")
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - policy import errors should surface as API errors.
        raise PolicyLoadError(f"Unable to import policy module {module_name!r}: {exc}") from exc


def _load_module_from_path(policy_path: str | None) -> ModuleType:
    path = _resolve_policy_path(policy_path)
    module_name = f"user_policy_{abs(hash((str(path), path.stat().st_mtime_ns)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PolicyLoadError(f"Unable to import policy file {path}.")

    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(POLICY_GLOBALS)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - policy import errors should surface as API errors.
        sys.modules.pop(module_name, None)
        raise PolicyLoadError(f"Unable to execute policy file {path}: {exc}") from exc
    return module


def _load_module_from_source(policy_source: str | None) -> ModuleType:
    if not policy_source or not policy_source.strip():
        raise PolicyLoadError("Missing policy_source.")

    module_name = f"user_policy_source_{abs(hash(policy_source))}"
    module = ModuleType(module_name)
    module.__dict__["__builtins__"] = __builtins__
    module.__dict__.update(POLICY_GLOBALS)
    sys.modules[module_name] = module
    try:
        exec(compile(policy_source, f"<{module_name}>", "exec"), module.__dict__)  # noqa: S102 - the runner executes user code.
    except Exception as exc:  # noqa: BLE001 - policy source errors should surface as API errors.
        sys.modules.pop(module_name, None)
        raise PolicyLoadError(f"Unable to execute policy source: {exc}") from exc
    return module


def _resolve_policy_path(policy_path: str | None) -> Path:
    if not policy_path:
        raise PolicyLoadError("Missing policy_path.")

    requested = Path(policy_path).expanduser()
    candidates: list[Path]
    if requested.is_absolute():
        candidates = [requested]
    else:
        candidates = [
            Path.cwd() / requested,
            PROJECT_ROOT / requested,
            PROJECT_ROOT / "samples" / "policies" / requested,
        ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.suffix == ".py":
            return resolved

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise PolicyLoadError(f"Policy file not found. Searched: {searched}")
