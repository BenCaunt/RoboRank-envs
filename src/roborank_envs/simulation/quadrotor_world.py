from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

from roborank_envs.models import RobotSpec
from roborank_envs.simulation.mujoco_world import REPLAY_RENDER_HEIGHT, REPLAY_RENDER_WIDTH, _rgb_to_png_bytes


@dataclass(frozen=True)
class GateVisual:
    id: str
    x: float
    y: float
    z: float
    yaw: float
    width_m: float
    height_m: float


@dataclass
class QuadrotorMujocoWorld:
    available: bool
    backend_name: str
    model: Any = None
    data: Any = None
    mujoco: Any = None
    drone_qpos_addr: int | None = None
    renderer: Any = None
    render_width: int = REPLAY_RENDER_WIDTH
    render_height: int = REPLAY_RENDER_HEIGHT
    render_camera: str = "overview"
    render_cameras: tuple[str, ...] = ("overview", "front_camera")
    timestep: float = 0.005

    @classmethod
    def create(cls, *, gates: list[GateVisual], bounds: dict[str, float], robot: RobotSpec) -> "QuadrotorMujocoWorld":
        try:
            import mujoco  # type: ignore[import-not-found]
        except Exception:
            return cls(available=False, backend_name="mujoco_unavailable_analytic_quadrotor")

        xml = _build_mjcf(gates=gates, bounds=bounds, robot=robot)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_free")
        return cls(
            available=True,
            backend_name="mujoco_visualization",
            model=model,
            data=data,
            mujoco=mujoco,
            drone_qpos_addr=int(model.jnt_qposadr[joint_id]),
            timestep=float(model.opt.timestep),
            render_width=REPLAY_RENDER_WIDTH,
            render_height=REPLAY_RENDER_HEIGHT,
            render_cameras=("overview", *(camera.name for camera in robot.cameras)),
        )

    def reset_drone_pose(self, *, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        self.set_drone_pose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)

    def set_drone_pose(self, *, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        if not self.available:
            return
        if self.drone_qpos_addr is None:
            raise RuntimeError("Quadrotor MuJoCo joint is not initialized.")

        quat = _euler_to_quat(roll=roll, pitch=pitch, yaw=yaw)
        addr = self.drone_qpos_addr
        self.data.qpos[addr : addr + 3] = [x, y, z]
        self.data.qpos[addr + 3 : addr + 7] = quat
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

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


def _build_mjcf(*, gates: list[GateVisual], bounds: dict[str, float], robot: RobotSpec) -> str:
    gates_xml = "\n".join(_gate_mjcf(gate) for gate in gates)
    drone_xml = _drone_mjcf(robot=robot)
    x_mid = (float(bounds["x_min"]) + float(bounds["x_max"])) / 2.0
    y_mid = (float(bounds["y_min"]) + float(bounds["y_max"])) / 2.0
    x_half = (float(bounds["x_max"]) - float(bounds["x_min"])) / 2.0
    y_half = (float(bounds["y_max"]) - float(bounds["y_min"])) / 2.0
    return f"""
<mujoco model="quadrotor_gate_sequence">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="{REPLAY_RENDER_WIDTH}" offheight="{REPLAY_RENDER_HEIGHT}"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.88 0.9 0.92" rgb2="0.72 0.78 0.8"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="7 4" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="1.5 -3.5 4.5" dir="-0.2 0.45 -1" diffuse="0.9 0.9 0.86"/>
    <geom name="floor" type="plane" pos="{x_mid} {y_mid} 0" size="{x_half} {y_half} 0.05" material="floor_mat" contype="0" conaffinity="0"/>
    <camera name="overview" pos="2.25 -4.35 2.25" xyaxes="1 0 0 0 0.42 0.91" fovy="52"/>
    {gates_xml}
    {drone_xml}
  </worldbody>
</mujoco>
"""


def _gate_mjcf(gate: GateVisual) -> str:
    tube = 0.045
    half_width = gate.width_m / 2.0
    half_height = gate.height_m / 2.0
    return f"""
    <body name="{gate.id}" pos="{gate.x} {gate.y} {gate.z}" euler="0 0 {gate.yaw}">
      <geom name="{gate.id}_top" type="box" size="{tube} {half_width + tube} {tube}" pos="0 0 {half_height}" rgba="0.95 0.62 0.12 1" contype="0" conaffinity="0"/>
      <geom name="{gate.id}_bottom" type="box" size="{tube} {half_width + tube} {tube}" pos="0 0 {-half_height}" rgba="0.95 0.62 0.12 1" contype="0" conaffinity="0"/>
      <geom name="{gate.id}_left" type="box" size="{tube} {tube} {half_height + tube}" pos="0 {half_width} 0" rgba="0.95 0.62 0.12 1" contype="0" conaffinity="0"/>
      <geom name="{gate.id}_right" type="box" size="{tube} {tube} {half_height + tube}" pos="0 {-half_width} 0" rgba="0.95 0.62 0.12 1" contype="0" conaffinity="0"/>
    </body>
"""


def _drone_mjcf(*, robot: RobotSpec) -> str:
    arm = float(getattr(robot, "arm_length_m", 0.225))
    body_x = 0.09
    body_y = 0.06
    body_z = robot.height_m / 2.0
    motor_radius = 0.04
    camera_xml = "\n".join(
        (
            f'<camera name="{camera.name}" pos="{camera.xyz_m["x"]} {camera.xyz_m["y"]} {camera.xyz_m["z"]}" '
            f'xyaxes="{camera.xyaxes}" fovy="{camera.fov_y_deg}"/>'
        )
        for camera in robot.cameras
    )
    return f"""
    <body name="drone" pos="0 0 1.0">
      <freejoint name="drone_free"/>
      <geom name="drone_body" type="box" size="{body_x} {body_y} {body_z}" rgba="0.1 0.22 0.34 1" contype="0" conaffinity="0" mass="{robot.mass_kg or 1.0}"/>
      <geom name="arm_x" type="capsule" fromto="{-arm} 0 0 {arm} 0 0" size="0.012" rgba="0.06 0.08 0.1 1" contype="0" conaffinity="0"/>
      <geom name="arm_y" type="capsule" fromto="0 {-arm} 0 0 {arm} 0" size="0.012" rgba="0.06 0.08 0.1 1" contype="0" conaffinity="0"/>
      <geom name="front_mark" type="box" size="0.026 0.034 0.011" pos="{body_x + 0.018} 0 0" rgba="0.15 0.75 0.95 1" contype="0" conaffinity="0"/>
      <geom name="motor_fl" type="cylinder" size="{motor_radius} 0.012" pos="{arm} {arm} 0" rgba="0.02 0.025 0.03 1" contype="0" conaffinity="0"/>
      <geom name="motor_fr" type="cylinder" size="{motor_radius} 0.012" pos="{arm} {-arm} 0" rgba="0.02 0.025 0.03 1" contype="0" conaffinity="0"/>
      <geom name="motor_rl" type="cylinder" size="{motor_radius} 0.012" pos="{-arm} {arm} 0" rgba="0.02 0.025 0.03 1" contype="0" conaffinity="0"/>
      <geom name="motor_rr" type="cylinder" size="{motor_radius} 0.012" pos="{-arm} {-arm} 0" rgba="0.02 0.025 0.03 1" contype="0" conaffinity="0"/>
      {camera_xml}
    </body>
"""


def _euler_to_quat(*, roll: float, pitch: float, yaw: float) -> list[float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return [
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ]
