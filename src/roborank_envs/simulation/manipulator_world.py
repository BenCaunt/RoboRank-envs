from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

from roborank_envs.models import ManipulatorSpec
from roborank_envs.simulation.mujoco_world import REPLAY_RENDER_HEIGHT, REPLAY_RENDER_WIDTH, _rgb_to_png_bytes


@dataclass(frozen=True)
class TargetVisual:
    x: float
    y: float
    z: float


@dataclass
class ManipulatorMujocoWorld:
    available: bool
    backend_name: str
    model: Any = None
    data: Any = None
    mujoco: Any = None
    qpos_addrs: dict[str, int] | None = None
    renderer: Any = None
    render_width: int = REPLAY_RENDER_WIDTH
    render_height: int = REPLAY_RENDER_HEIGHT
    render_cameras: tuple[str, ...] = ("overview", "top")
    timestep: float = 0.005

    @classmethod
    def create(cls, *, robot: ManipulatorSpec, targets: list[TargetVisual]) -> "ManipulatorMujocoWorld":
        try:
            import mujoco  # type: ignore[import-not-found]
        except Exception:
            return cls(available=False, backend_name="mujoco_unavailable_analytic_ik")

        xml = _build_mjcf(robot=robot, targets=targets)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        joint_names = [f"link_{index}_free" for index in range(4)] + ["target_free"]
        qpos_addrs: dict[str, int] = {}
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addrs[name] = int(model.jnt_qposadr[joint_id])
        cameras = ("overview", "side") if robot.mechanism == "turret_2link" else ("overview", "top")
        return cls(
            available=True,
            backend_name="mujoco_manipulator_visual",
            model=model,
            data=data,
            mujoco=mujoco,
            qpos_addrs=qpos_addrs,
            timestep=float(model.opt.timestep),
            render_cameras=cameras,
        )

    def update(
        self,
        *,
        segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
        target: tuple[float, float, float],
    ) -> None:
        if not self.available or self.qpos_addrs is None:
            return
        for index in range(4):
            segment = segments[index] if index < len(segments) else None
            if segment is None:
                self._set_free_body(f"link_{index}_free", center=(0.0, 0.0, -10.0), quat=(1.0, 0.0, 0.0, 0.0))
                continue
            start, end = segment
            center = tuple((start[axis] + end[axis]) / 2.0 for axis in range(3))
            quat = _quat_from_x_axis(start=start, end=end)
            self._set_free_body(f"link_{index}_free", center=center, quat=quat)

        self._set_free_body("target_free", center=target, quat=(1.0, 0.0, 0.0, 0.0))
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def render_data_url(self, *, camera: str) -> str:
        if not self.available:
            raise RuntimeError("MuJoCo manipulator world is not available.")
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(
                self.model,
                height=self.render_height,
                width=self.render_width,
            )

        self.renderer.update_scene(self.data, camera=camera)
        pixels = self.renderer.render()
        png_bytes = _rgb_to_png_bytes(pixels)
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _set_free_body(self, name: str, *, center: tuple[float, float, float], quat: tuple[float, float, float, float]) -> None:
        assert self.qpos_addrs is not None
        addr = self.qpos_addrs[name]
        self.data.qpos[addr : addr + 3] = center
        self.data.qpos[addr + 3 : addr + 7] = quat


def _build_mjcf(*, robot: ManipulatorSpec, targets: list[TargetVisual]) -> str:
    l1 = float(robot.link_lengths_m[0])
    l2 = float(robot.link_lengths_m[1])
    lengths = [l1, l2, l1, l2]
    link_xml = "\n".join(_link_body(index=index, length=length, visible=index < 2 or robot.mechanism == "five_bar_scara") for index, length in enumerate(lengths))
    target_path_xml = "\n".join(_target_marker(index=index, target=target) for index, target in enumerate(targets[:96]))
    base_xml = _base_xml(robot)
    camera_xml = _camera_xml(robot)
    floor_size = max(1.0, robot.workspace_radius_m + 0.18)
    return f"""
<mujoco model="{robot.model}">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="{REPLAY_RENDER_WIDTH}" offheight="{REPLAY_RENDER_HEIGHT}"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.88 0.90 0.92" rgb2="0.74 0.78 0.80"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="4 4" reflectance="0.07"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="0.6 -1.4 2.2" dir="-0.2 0.45 -1" diffuse="0.88 0.88 0.84"/>
    <geom name="floor" type="plane" size="{floor_size} {floor_size} 0.04" material="floor_mat" contype="0" conaffinity="0"/>
    {base_xml}
    {target_path_xml}
    <body name="current_target" pos="0 0 0.04">
      <freejoint name="target_free"/>
      <geom name="target" type="sphere" size="0.024" rgba="0.85 0.18 0.12 1" contype="0" conaffinity="0"/>
    </body>
    {link_xml}
    {camera_xml}
  </worldbody>
</mujoco>
"""


def _link_body(*, index: int, length: float, visible: bool) -> str:
    colors = [
        "0.10 0.32 0.68 1",
        "0.08 0.48 0.42 1",
        "0.76 0.38 0.12 1",
        "0.68 0.20 0.18 1",
    ]
    rgba = colors[index] if visible else "0 0 0 0"
    radius = 0.016 if index < 2 else 0.013
    joint_radius = 0.025 if visible else 0.001
    return f"""
    <body name="link_{index}" pos="0 0 -10">
      <freejoint name="link_{index}_free"/>
      <geom name="link_{index}_bar" type="capsule" fromto="{-length / 2} 0 0 {length / 2} 0 0" size="{radius}" rgba="{rgba}" contype="0" conaffinity="0"/>
      <geom name="link_{index}_start" type="sphere" pos="{-length / 2} 0 0" size="{joint_radius}" rgba="{rgba}" contype="0" conaffinity="0"/>
      <geom name="link_{index}_end" type="sphere" pos="{length / 2} 0 0" size="{joint_radius}" rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>
"""


def _target_marker(*, index: int, target: TargetVisual) -> str:
    return (
        f'<geom name="target_path_{index}" type="sphere" pos="{target.x} {target.y} {target.z}" '
        'size="0.008" rgba="0.85 0.18 0.12 0.45" contype="0" conaffinity="0"/>'
    )


def _base_xml(robot: ManipulatorSpec) -> str:
    if robot.mechanism == "five_bar_scara":
        spacing = float(robot.base_spacing_m or 0.36)
        return f"""
    <geom name="left_base" type="cylinder" pos="{-spacing / 2} 0 0.025" size="0.040 0.025" rgba="0.08 0.10 0.13 1" contype="0" conaffinity="0"/>
    <geom name="right_base" type="cylinder" pos="{spacing / 2} 0 0.025" size="0.040 0.025" rgba="0.08 0.10 0.13 1" contype="0" conaffinity="0"/>
    <geom name="base_bar" type="box" pos="0 0 0.012" size="{spacing / 2 + 0.05} 0.030 0.012" rgba="0.18 0.22 0.27 1" contype="0" conaffinity="0"/>
"""
    return f"""
    <geom name="base" type="cylinder" pos="0 0 {max(0.03, robot.base_height_m / 2)}" size="0.055 {max(0.03, robot.base_height_m / 2)}" rgba="0.08 0.10 0.13 1" contype="0" conaffinity="0"/>
"""


def _camera_xml(robot: ManipulatorSpec) -> str:
    if robot.mechanism == "turret_2link":
        return """
    <camera name="overview" pos="0.95 -1.20 0.82" xyaxes="0.78 0.62 0 -0.22 0.28 0.94" fovy="46"/>
    <camera name="side" pos="1.25 -0.10 0.64" xyaxes="0 1 0 -0.34 0 0.94" fovy="42"/>
"""
    return """
    <camera name="overview" pos="0.0 -1.15 1.08" xyaxes="1 0 0 0 0.58 0.82" fovy="44"/>
    <camera name="top" pos="0 0 1.42" xyaxes="1 0 0 0 1 0" fovy="44"/>
"""


def _quat_from_x_axis(
    *,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    vector = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if length <= 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    vx, vy, vz = (component / length for component in vector)
    dot = max(-1.0, min(1.0, vx))
    axis = (0.0, -vz, vy)
    axis_norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if axis_norm <= 1e-9:
        if dot >= 0.0:
            return (1.0, 0.0, 0.0, 0.0)
        return (0.0, 0.0, 0.0, 1.0)
    angle = math.atan2(axis_norm, dot)
    scale = math.sin(angle / 2.0) / axis_norm
    return (math.cos(angle / 2.0), axis[0] * scale, axis[1] * scale, axis[2] * scale)
