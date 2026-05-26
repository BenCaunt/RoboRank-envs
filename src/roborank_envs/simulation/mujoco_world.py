from __future__ import annotations

import base64
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any

from roborank_envs.models import Obstacle, Pose2D, RobotSpec, Target

REPLAY_RENDER_WIDTH = 768
REPLAY_RENDER_HEIGHT = 480


@dataclass(frozen=True)
class MujocoContact:
    kind: str
    object_id: str
    penetration_m: float


@dataclass
class MujocoWorld:
    available: bool
    backend_name: str
    model: Any = None
    data: Any = None
    mujoco: Any = None
    robot_x_qpos_addr: int | None = None
    robot_y_qpos_addr: int | None = None
    robot_yaw_qpos_addr: int | None = None
    robot_x_dof_addr: int | None = None
    robot_y_dof_addr: int | None = None
    robot_yaw_dof_addr: int | None = None
    robot_geom_id: int | None = None
    obstacle_geom_ids: dict[int, str] | None = None
    wall_geom_ids: dict[int, str] | None = None
    timestep: float = 0.005
    force_gain: float = 75.0
    torque_gain: float = 9.0
    max_force: float = 120.0
    max_torque: float = 24.0
    renderer: Any = None
    render_width: int = REPLAY_RENDER_WIDTH
    render_height: int = REPLAY_RENDER_HEIGHT
    render_camera: str = "overview"
    render_cameras: tuple[str, ...] = ("overview", "front_camera")

    @classmethod
    def create(
        cls,
        *,
        target: Target,
        obstacles: list[Obstacle],
        bounds: dict[str, float],
        robot: RobotSpec,
        target_yaw: float = 0.0,
        target_kind: str = "target_disk",
    ) -> "MujocoWorld":
        try:
            import mujoco  # type: ignore[import-not-found]
        except Exception:
            return cls(available=False, backend_name="mujoco_compatible_kinematic")

        xml = _build_mjcf(
            target=target,
            obstacles=obstacles,
            bounds=bounds,
            robot=robot,
            target_yaw=target_yaw,
            target_kind=target_kind,
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        x_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_x")
        y_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_y")
        yaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_yaw")
        robot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_geom")
        obstacle_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{obstacle.id}_geom"): obstacle.id
            for obstacle in obstacles
        }
        wall_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{wall_name}_geom"): wall_name
            for wall_name in ("left_wall", "right_wall", "bottom_wall", "top_wall")
        }
        return cls(
            available=True,
            backend_name="mujoco_physics",
            model=model,
            data=data,
            mujoco=mujoco,
            robot_x_qpos_addr=int(model.jnt_qposadr[x_joint_id]),
            robot_y_qpos_addr=int(model.jnt_qposadr[y_joint_id]),
            robot_yaw_qpos_addr=int(model.jnt_qposadr[yaw_joint_id]),
            robot_x_dof_addr=int(model.jnt_dofadr[x_joint_id]),
            robot_y_dof_addr=int(model.jnt_dofadr[y_joint_id]),
            robot_yaw_dof_addr=int(model.jnt_dofadr[yaw_joint_id]),
            robot_geom_id=int(robot_geom_id),
            obstacle_geom_ids=obstacle_geom_ids,
            wall_geom_ids=wall_geom_ids,
            timestep=float(model.opt.timestep),
            render_width=REPLAY_RENDER_WIDTH,
            render_height=REPLAY_RENDER_HEIGHT,
            render_cameras=("overview", *(camera.name for camera in robot.cameras)),
        )

    def reset_robot_pose(self, pose: Pose2D) -> None:
        if not self.available:
            return

        self.data.qpos[self.robot_x_qpos_addr] = pose.x
        self.data.qpos[self.robot_y_qpos_addr] = pose.y
        self.data.qpos[self.robot_yaw_qpos_addr] = pose.yaw
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def pose(self) -> Pose2D:
        if not self.available:
            raise RuntimeError("MuJoCo world is not available.")
        return Pose2D(
            x=float(self.data.qpos[self.robot_x_qpos_addr]),
            y=float(self.data.qpos[self.robot_y_qpos_addr]),
            yaw=_wrap_angle(float(self.data.qpos[self.robot_yaw_qpos_addr])),
        )

    def step_diff_drive(self, *, left: float, right: float, wheel_base: float, dt: float) -> Pose2D:
        if not self.available:
            raise RuntimeError("MuJoCo world is not available.")

        linear_velocity = 0.5 * (left + right)
        angular_velocity = (right - left) / wheel_base
        substeps = max(1, round(dt / self.timestep))

        for _ in range(substeps):
            pose = self.pose()
            target_vx = linear_velocity * math.cos(pose.yaw)
            target_vy = linear_velocity * math.sin(pose.yaw)
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.robot_x_dof_addr] = _clamp(
                self.force_gain * (target_vx - float(self.data.qvel[self.robot_x_dof_addr])),
                -self.max_force,
                self.max_force,
            )
            self.data.qfrc_applied[self.robot_y_dof_addr] = _clamp(
                self.force_gain * (target_vy - float(self.data.qvel[self.robot_y_dof_addr])),
                -self.max_force,
                self.max_force,
            )
            self.data.qfrc_applied[self.robot_yaw_dof_addr] = _clamp(
                self.torque_gain * (angular_velocity - float(self.data.qvel[self.robot_yaw_dof_addr])),
                -self.max_torque,
                self.max_torque,
            )
            self.mujoco.mj_step(self.model, self.data)

        self.data.qfrc_applied[:] = 0.0
        return self.pose()

    def first_contact(self) -> MujocoContact | None:
        if (
            not self.available
            or self.robot_geom_id is None
            or self.obstacle_geom_ids is None
            or self.wall_geom_ids is None
        ):
            return None

        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_ids = {int(contact.geom1), int(contact.geom2)}
            if self.robot_geom_id not in geom_ids:
                continue

            other_geom_id = next(geom_id for geom_id in geom_ids if geom_id != self.robot_geom_id)
            penetration = max(0.0, -float(contact.dist))
            if other_geom_id in self.obstacle_geom_ids:
                return MujocoContact(
                    kind="obstacle",
                    object_id=self.obstacle_geom_ids[other_geom_id],
                    penetration_m=penetration,
                )
            if other_geom_id in self.wall_geom_ids:
                return MujocoContact(
                    kind="bounds",
                    object_id=self.wall_geom_ids[other_geom_id],
                    penetration_m=penetration,
                )
        return None

    def render_data_url(self, *, camera: str | None = None) -> str:
        if not self.available:
            raise RuntimeError("MuJoCo world is not available.")
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(
                self.model,
                height=self.render_height,
                width=self.render_width,
            )

        self.renderer.update_scene(self.data, camera=camera or self.render_camera)
        pixels = self.renderer.render()
        png_bytes = _rgb_to_png_bytes(pixels)
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def _build_mjcf(
    *,
    target: Target,
    obstacles: list[Obstacle],
    bounds: dict[str, float],
    robot: RobotSpec,
    target_yaw: float,
    target_kind: str,
) -> str:
    robot_height = robot.height_m
    ground_size_x = max(3.0, (float(bounds["x_max"]) - float(bounds["x_min"])) / 2 + 0.35)
    ground_size_y = max(3.0, (float(bounds["y_max"]) - float(bounds["y_min"])) / 2 + 0.35)
    obstacle_xml = "\n".join(
        (
            f'<body name="{obstacle.id}" pos="{obstacle.x} {obstacle.y} {robot_height / 2}">'
            f'<geom name="{obstacle.id}_geom" type="cylinder" size="{obstacle.radius} {robot_height / 2}" '
            'rgba="0.8 0.2 0.2 1" contype="1" conaffinity="1"/>'
            "</body>"
        )
        for obstacle in obstacles
    )
    wall_xml = _build_wall_mjcf(bounds=bounds, robot_height=robot_height)
    robot_xml = _build_robot_mjcf(robot=robot)
    target_xml = _build_target_mjcf(target=target, target_yaw=target_yaw, target_kind=target_kind)
    return f"""
<mujoco model="diff_drive_navigation">
  <option timestep="0.005" gravity="0 0 0" integrator="RK4"/>
  <visual>
    <global offwidth="{REPLAY_RENDER_WIDTH}" offheight="{REPLAY_RENDER_HEIGHT}"/>
  </visual>
  <default>
    <geom solref="0.01 1" solimp="0.9 0.95 0.001" friction="1 0.1 0.1"/>
  </default>
  <asset>
    <texture name="grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.88 0.9 0.92" rgb2="0.74 0.78 0.82"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="6 5" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="0 -3 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="{ground_size_x} {ground_size_y} 0.05" material="grid_mat" contype="0" conaffinity="0"/>
    {target_xml}
    {obstacle_xml}
    {wall_xml}
    {robot_xml}
  </worldbody>
</mujoco>
"""


def _build_robot_mjcf(*, robot: RobotSpec) -> str:
    half_length = robot.footprint_length_m / 2
    half_width = robot.footprint_width_m / 2
    half_height = robot.height_m / 2
    wheel_y = robot.wheel_base_m / 2
    wheel_x = -0.015
    wheel_z = robot.wheel_radius_m
    wheel_half_width = robot.wheel_width_m / 2
    lidar = robot.lidars[0] if robot.lidars else None
    camera_xml = "\n".join(
        (
            f'<camera name="{camera.name}" pos="{camera.xyz_m["x"]} {camera.xyz_m["y"]} {camera.xyz_m["z"]}" '
            f'xyaxes="{camera.xyaxes}" fovy="{camera.fov_y_deg}"/>'
        )
        for camera in robot.cameras
    )
    lidar_xml = ""
    if lidar is not None:
        lidar_xml = f"""
      <site name="{lidar.frame}_site" pos="{lidar.xyz_m["x"]} {lidar.xyz_m["y"]} {lidar.xyz_m["z"]}" size="0.012" rgba="0.08 0.08 0.08 1"/>
      <geom name="{lidar.frame}_visual" type="cylinder" size="0.052 0.018" pos="{lidar.xyz_m["x"]} {lidar.xyz_m["y"]} {lidar.xyz_m["z"]}" rgba="0.05 0.08 0.1 1" contype="0" conaffinity="0" density="0"/>
"""

    return f"""
    <body name="robot" pos="0 0 0">
      <joint name="robot_x" type="slide" axis="1 0 0" damping="0.35"/>
      <joint name="robot_y" type="slide" axis="0 1 0" damping="0.35"/>
      <joint name="robot_yaw" type="hinge" axis="0 0 1" damping="0.08"/>
      <geom name="robot_geom" type="box" size="{half_length} {half_width} {half_height}" pos="0 0 {half_height}" rgba="0.1 0.35 0.95 1" contype="1" conaffinity="1" mass="8"/>
      <geom name="robot_front_panel" type="box" size="0.012 {half_width * 0.68} {half_height * 0.42}" pos="{half_length + 0.004} 0 {half_height * 1.16}" rgba="0.02 0.08 0.12 1" contype="0" conaffinity="0" density="0"/>
      <geom name="left_wheel_geom" type="cylinder" size="{robot.wheel_radius_m} {wheel_half_width}" pos="{wheel_x} {wheel_y} {wheel_z}" euler="1.5707963268 0 0" rgba="0.02 0.025 0.03 1" contype="0" conaffinity="0" density="0"/>
      <geom name="right_wheel_geom" type="cylinder" size="{robot.wheel_radius_m} {wheel_half_width}" pos="{wheel_x} {-wheel_y} {wheel_z}" euler="1.5707963268 0 0" rgba="0.02 0.025 0.03 1" contype="0" conaffinity="0" density="0"/>
      <camera name="overview" pos="-0.9 0 0.65" xyaxes="0 -1 0 0.37 0 0.93" fovy="58"/>
      {camera_xml}
      {lidar_xml}
    </body>
"""


def _build_target_mjcf(*, target: Target, target_yaw: float, target_kind: str) -> str:
    if target_kind == "charging_pad":
        return f"""
    <body name="target" pos="{target.x} {target.y} 0.012" euler="0 0 {target_yaw}">
      <geom name="target_geom" type="box" size="0.18 0.24 0.012" rgba="0.04 0.55 0.45 0.42" contype="0" conaffinity="0"/>
      <geom name="charger_marker" type="box" size="0.025 0.16 0.014" pos="-0.12 0 0.018" rgba="0.85 0.96 0.98 0.92" contype="0" conaffinity="0"/>
      <geom name="charger_contact_strip" type="box" size="0.018 0.20 0.016" pos="0.12 0 0.02" rgba="0.05 0.12 0.16 0.95" contype="0" conaffinity="0"/>
    </body>
"""

    return f"""
    <body name="target" pos="{target.x} {target.y} 0.01">
      <geom name="target_geom" type="cylinder" size="{target.radius} 0.01" rgba="0.1 0.7 0.2 0.35" contype="0" conaffinity="0"/>
    </body>
"""


def _build_wall_mjcf(*, bounds: dict[str, float], robot_height: float) -> str:
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["y_min"])
    y_max = float(bounds["y_max"])
    thickness = 0.08
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    half_width = (x_max - x_min) / 2 + thickness
    half_height = (y_max - y_min) / 2 + thickness
    wall_z = robot_height / 2
    return f"""
    <body name="left_wall" pos="{x_min - thickness / 2} {y_mid} {wall_z}">
      <geom name="left_wall_geom" type="box" size="{thickness / 2} {half_height} {robot_height / 2}" rgba="0.4 0.45 0.5 0.35" contype="1" conaffinity="1"/>
    </body>
    <body name="right_wall" pos="{x_max + thickness / 2} {y_mid} {wall_z}">
      <geom name="right_wall_geom" type="box" size="{thickness / 2} {half_height} {robot_height / 2}" rgba="0.4 0.45 0.5 0.35" contype="1" conaffinity="1"/>
    </body>
    <body name="bottom_wall" pos="{x_mid} {y_min - thickness / 2} {wall_z}">
      <geom name="bottom_wall_geom" type="box" size="{half_width} {thickness / 2} {robot_height / 2}" rgba="0.4 0.45 0.5 0.35" contype="1" conaffinity="1"/>
    </body>
    <body name="top_wall" pos="{x_mid} {y_max + thickness / 2} {wall_z}">
      <geom name="top_wall_geom" type="box" size="{half_width} {thickness / 2} {robot_height / 2}" rgba="0.4 0.45 0.5 0.35" contype="1" conaffinity="1"/>
    </body>
"""


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rgb_to_png_bytes(pixels: Any) -> bytes:
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("Expected RGB render output.")

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(data, checksum)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum & 0xFFFFFFFF)

    raw_rows = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw_rows, level=6)),
            chunk(b"IEND", b""),
        ]
    )
