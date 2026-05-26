from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass
from typing import Any

from roborank_envs.models import MotorStandSpec
from roborank_envs.simulation.mujoco_world import REPLAY_RENDER_HEIGHT, REPLAY_RENDER_WIDTH


@dataclass
class MotorStandWorld:
    available: bool
    backend_name: str
    model: Any = None
    data: Any = None
    mujoco: Any = None
    shaft_qpos_addr: int | None = None
    platen_qpos_addr: int | None = None
    renderer: Any = None
    timestep: float = 0.005
    render_width: int = REPLAY_RENDER_WIDTH
    render_height: int = REPLAY_RENDER_HEIGHT
    render_cameras: tuple[str, ...] = ("overview", "scale_closeup")

    @classmethod
    def create(cls, *, fixture: MotorStandSpec) -> "MotorStandWorld":
        try:
            import mujoco  # type: ignore[import-not-found]
        except Exception:
            return cls(available=False, backend_name="mujoco_compatible_motor_fallback")

        xml = _build_mjcf(fixture=fixture)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        shaft_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_hinge")
        platen_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scale_platen_z")
        return cls(
            available=True,
            backend_name="mujoco_motor_fixture",
            model=model,
            data=data,
            mujoco=mujoco,
            shaft_qpos_addr=int(model.jnt_qposadr[shaft_joint_id]),
            platen_qpos_addr=int(model.jnt_qposadr[platen_joint_id]),
            timestep=float(model.opt.timestep),
        )

    def update(self, *, shaft_angle_rad: float, platen_deflection_m: float) -> None:
        if not self.available:
            return
        if self.shaft_qpos_addr is None or self.platen_qpos_addr is None:
            return

        self.data.qpos[self.shaft_qpos_addr] = shaft_angle_rad
        self.data.qpos[self.platen_qpos_addr] = -min(0.012, max(0.0, platen_deflection_m))
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def render_data_url(self, *, camera: str) -> str:
        if not self.available:
            raise RuntimeError("MuJoCo motor fixture is not available.")
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


def _build_mjcf(*, fixture: MotorStandSpec) -> str:
    shaft_length = fixture.shaft_length_m
    shaft_radius = fixture.shaft_radius_m
    half_base_x = fixture.base_length_m / 2
    half_base_y = fixture.base_width_m / 2
    half_base_z = fixture.base_height_m / 2
    half_plate_x = fixture.scale_plate_width_m / 2
    half_plate_y = fixture.scale_plate_depth_m / 2
    hinge_x = -0.075
    hinge_z = 0.175
    scale_x = hinge_x + shaft_length
    press_foot_length = 0.086

    return f"""
<mujoco model="motor_stand_scale">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="{REPLAY_RENDER_WIDTH}" offheight="{REPLAY_RENDER_HEIGHT}"/>
  </visual>
  <default>
    <geom solref="0.01 1" solimp="0.9 0.95 0.001" friction="1 0.1 0.1"/>
  </default>
  <asset>
    <texture name="bench_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.84 0.86 0.82" rgb2="0.66 0.69 0.66"/>
    <material name="bench_mat" texture="bench_tex" texrepeat="5 4" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="-0.4 -0.8 1.2" dir="0.2 0.4 -1" diffuse="0.9 0.9 0.85"/>
    <geom name="bench" type="plane" size="0.55 0.38 0.03" material="bench_mat" contype="0" conaffinity="0"/>
    <body name="base" pos="0 0 {half_base_z}">
      <geom name="base_geom" type="box" size="{half_base_x} {half_base_y} {half_base_z}" rgba="0.18 0.24 0.28 1" contype="0" conaffinity="0"/>
      <geom name="base_front_label" type="box" size="0.055 0.002 0.012" pos="-0.03 {-half_base_y - 0.003} 0.004" rgba="0.02 0.06 0.08 1" contype="0" conaffinity="0"/>
    </body>
    <body name="motor_block" pos="{hinge_x - 0.035} 0 {hinge_z}">
      <geom name="motor_mount" type="box" size="0.018 0.058 0.055" pos="-0.024 0 -0.018" rgba="0.12 0.16 0.18 1" contype="0" conaffinity="0"/>
      <geom name="motor_can" type="cylinder" size="0.042 0.042" euler="0 1.5707963268 0" rgba="0.05 0.10 0.13 1" contype="0" conaffinity="0"/>
      <geom name="motor_face" type="cylinder" size="0.046 0.006" pos="0.043 0 0" euler="0 1.5707963268 0" rgba="0.85 0.72 0.35 1" contype="0" conaffinity="0"/>
    </body>
    <body name="shaft" pos="{hinge_x} 0 {hinge_z}">
      <joint name="shaft_hinge" type="hinge" axis="0 1 0" limited="true" range="-0.55 0.18" damping="0.01"/>
      <geom name="shaft_geom" type="capsule" fromto="0 0 0 {shaft_length} 0 0" size="{shaft_radius}" rgba="0.76 0.78 0.78 1" contype="0" conaffinity="0"/>
      <geom name="press_tip" type="capsule" fromto="{shaft_length} 0 0 {shaft_length} 0 {-press_foot_length}" size="0.009" rgba="0.04 0.08 0.10 1" contype="0" conaffinity="0"/>
      <geom name="press_pad" type="sphere" size="0.018" pos="{shaft_length} 0 {-press_foot_length}" rgba="0.04 0.08 0.10 1" contype="0" conaffinity="0"/>
    </body>
    <body name="scale_body" pos="{scale_x} 0 0.048">
      <geom name="scale_case" type="box" size="0.082 0.072 0.020" rgba="0.86 0.88 0.90 1" contype="0" conaffinity="0"/>
      <geom name="scale_display" type="box" size="0.038 0.004 0.010" pos="-0.006 -0.075 0.006" rgba="0.04 0.10 0.09 1" contype="0" conaffinity="0"/>
      <body name="scale_platen" pos="0 0 0.032">
        <joint name="scale_platen_z" type="slide" axis="0 0 1" limited="true" range="-0.012 0.002" damping="0.02"/>
        <geom name="scale_plate" type="box" size="{half_plate_x} {half_plate_y} 0.006" rgba="0.48 0.58 0.64 1" contype="0" conaffinity="0"/>
      </body>
    </body>
    <camera name="overview" pos="0.18 -0.55 0.38" xyaxes="1 0 0 0 0.56 0.83" fovy="45"/>
    <camera name="scale_closeup" pos="0.14 -0.32 0.20" xyaxes="1 0 0 0 0.42 0.91" fovy="34"/>
  </worldbody>
</mujoco>
"""


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
