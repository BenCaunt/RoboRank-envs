from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass
from typing import Any

from roborank_envs.models import CartPoleSpec
from roborank_envs.simulation.mujoco_world import REPLAY_RENDER_HEIGHT, REPLAY_RENDER_WIDTH


@dataclass
class CartPoleWorld:
    available: bool
    backend_name: str
    model: Any = None
    data: Any = None
    mujoco: Any = None
    cart_qpos_addr: int | None = None
    pole_qpos_addr: int | None = None
    renderer: Any = None
    timestep: float = 0.005
    render_width: int = REPLAY_RENDER_WIDTH
    render_height: int = REPLAY_RENDER_HEIGHT
    render_cameras: tuple[str, ...] = ("overview", "side")

    @classmethod
    def create(cls, *, plant: CartPoleSpec) -> "CartPoleWorld":
        try:
            import mujoco  # type: ignore[import-not-found]
        except Exception:
            return cls(available=False, backend_name="mujoco_compatible_cart_pole_fallback")

        xml = _build_mjcf(plant=plant)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        cart_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_x")
        pole_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pole_hinge")
        return cls(
            available=True,
            backend_name="mujoco_cart_pole_visual",
            model=model,
            data=data,
            mujoco=mujoco,
            cart_qpos_addr=int(model.jnt_qposadr[cart_joint_id]),
            pole_qpos_addr=int(model.jnt_qposadr[pole_joint_id]),
            timestep=float(model.opt.timestep),
        )

    def update(self, *, cart_position_m: float, pole_angle_rad: float) -> None:
        if not self.available or self.cart_qpos_addr is None or self.pole_qpos_addr is None:
            return

        self.data.qpos[self.cart_qpos_addr] = cart_position_m
        self.data.qpos[self.pole_qpos_addr] = pole_angle_rad
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def render_data_url(self, *, camera: str) -> str:
        if not self.available:
            raise RuntimeError("MuJoCo cart-pole world is not available.")
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


def _build_mjcf(*, plant: CartPoleSpec) -> str:
    track = plant.track_half_width_m
    cart_half_x = plant.cart_width_m / 2
    cart_half_y = 0.09
    cart_half_z = plant.cart_height_m / 2
    cart_z = cart_half_z + 0.035
    pivot_z = cart_z + cart_half_z + 0.012
    pole_length = plant.pole_length_m
    rail_z = 0.028
    return f"""
<mujoco model="cart_pole">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="{REPLAY_RENDER_WIDTH}" offheight="{REPLAY_RENDER_HEIGHT}"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.88 0.90 0.92" rgb2="0.70 0.74 0.78"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="5 3" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="0 -3 4" dir="0 0 -1" diffuse="0.85 0.85 0.82"/>
    <geom name="floor" type="plane" size="{track + 0.45} 0.8 0.04" material="floor_mat" contype="0" conaffinity="0"/>
    <geom name="rail" type="box" pos="0 0 {rail_z}" size="{track} 0.035 0.018" rgba="0.22 0.27 0.32 1" contype="0" conaffinity="0"/>
    <geom name="left_stop" type="box" pos="{-track} 0 {rail_z + 0.06}" size="0.025 0.12 0.08" rgba="0.72 0.18 0.16 1" contype="0" conaffinity="0"/>
    <geom name="right_stop" type="box" pos="{track} 0 {rail_z + 0.06}" size="0.025 0.12 0.08" rgba="0.72 0.18 0.16 1" contype="0" conaffinity="0"/>
    <body name="cart" pos="0 0 {cart_z}">
      <joint name="cart_x" type="slide" axis="1 0 0" limited="true" range="{-track} {track}" damping="0.02"/>
      <geom name="cart_body" type="box" size="{cart_half_x} {cart_half_y} {cart_half_z}" rgba="0.10 0.32 0.68 1" contype="0" conaffinity="0"/>
      <geom name="left_wheel" type="cylinder" pos="{-cart_half_x * 0.58} -0.102 {-cart_half_z}" size="0.038 0.018" euler="1.5707963268 0 0" rgba="0.03 0.04 0.05 1" contype="0" conaffinity="0"/>
      <geom name="right_wheel" type="cylinder" pos="{cart_half_x * 0.58} -0.102 {-cart_half_z}" size="0.038 0.018" euler="1.5707963268 0 0" rgba="0.03 0.04 0.05 1" contype="0" conaffinity="0"/>
      <body name="pole" pos="0 0 {pivot_z - cart_z}">
        <joint name="pole_hinge" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 {pole_length}" size="{plant.pole_radius_m}" rgba="0.88 0.56 0.18 1" contype="0" conaffinity="0"/>
        <geom name="pivot" type="sphere" size="0.035" rgba="0.05 0.08 0.10 1" contype="0" conaffinity="0"/>
        <geom name="tip" type="sphere" pos="0 0 {pole_length}" size="0.026" rgba="0.85 0.20 0.15 1" contype="0" conaffinity="0"/>
      </body>
    </body>
    <camera name="overview" pos="0 -4.1 1.65" xyaxes="1 0 0 0 0.38 0.92" fovy="38"/>
    <camera name="side" pos="0 -2.9 0.85" xyaxes="1 0 0 0 0.16 0.99" fovy="42"/>
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
