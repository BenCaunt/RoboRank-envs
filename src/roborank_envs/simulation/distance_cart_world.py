from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass
from typing import Any

from roborank_envs.models import ProfiledCartSpec
from roborank_envs.simulation.mujoco_world import REPLAY_RENDER_HEIGHT, REPLAY_RENDER_WIDTH


@dataclass
class DistanceCartWorld:
    available: bool
    backend_name: str
    model: Any = None
    data: Any = None
    mujoco: Any = None
    cart_qpos_addr: int | None = None
    renderer: Any = None
    timestep: float = 0.005
    render_width: int = REPLAY_RENDER_WIDTH
    render_height: int = REPLAY_RENDER_HEIGHT
    render_cameras: tuple[str, ...] = ("overview", "side")

    @classmethod
    def create(cls, *, cart: ProfiledCartSpec, wall_position_m: float) -> "DistanceCartWorld":
        try:
            import mujoco  # type: ignore[import-not-found]
        except Exception:
            return cls(available=False, backend_name="mujoco_compatible_distance_cart_fallback")

        xml = _build_mjcf(cart=cart, wall_position_m=wall_position_m)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        cart_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_x")
        return cls(
            available=True,
            backend_name="mujoco_distance_cart_visual",
            model=model,
            data=data,
            mujoco=mujoco,
            cart_qpos_addr=int(model.jnt_qposadr[cart_joint_id]),
            timestep=float(model.opt.timestep),
        )

    def update(self, *, cart_position_m: float) -> None:
        if not self.available or self.cart_qpos_addr is None:
            return

        self.data.qpos[self.cart_qpos_addr] = cart_position_m
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def render_data_url(self, *, camera: str) -> str:
        if not self.available:
            raise RuntimeError("MuJoCo distance-cart world is not available.")
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


def _build_mjcf(*, cart: ProfiledCartSpec, wall_position_m: float) -> str:
    track = cart.track_half_width_m
    cart_half_x = cart.cart_width_m / 2
    cart_half_y = 0.12
    cart_half_z = cart.cart_height_m / 2
    cart_z = cart_half_z + 0.04
    rail_z = 0.03
    wall_height = 0.56
    wall_z = rail_z + wall_height / 2
    return f"""
<mujoco model="distance_cart_1d">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="{REPLAY_RENDER_WIDTH}" offheight="{REPLAY_RENDER_HEIGHT}"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.88 0.90 0.92" rgb2="0.70 0.74 0.78"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 2" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="0 -3.2 4" dir="0 0 -1" diffuse="0.85 0.85 0.82"/>
    <geom name="floor" type="plane" size="{track + 0.5} 0.75 0.04" material="floor_mat" contype="0" conaffinity="0"/>
    <geom name="rail" type="box" pos="0 0 {rail_z}" size="{track} 0.035 0.018" rgba="0.22 0.27 0.32 1" contype="0" conaffinity="0"/>
    <geom name="left_stop" type="box" pos="{-track} 0 {rail_z + 0.055}" size="0.026 0.12 0.075" rgba="0.72 0.18 0.16 1" contype="0" conaffinity="0"/>
    <geom name="wall" type="box" pos="{wall_position_m} 0 {wall_z}" size="0.045 0.42 {wall_height / 2}" rgba="0.72 0.18 0.16 1" contype="0" conaffinity="0"/>
    <geom name="sensor_beam" type="box" pos="{wall_position_m / 2} -0.18 {cart_z + 0.025}" size="{max(0.05, wall_position_m / 2)} 0.006 0.006" rgba="0.10 0.48 0.64 0.45" contype="0" conaffinity="0"/>
    <body name="cart" pos="0 0 {cart_z}">
      <joint name="cart_x" type="slide" axis="1 0 0" limited="true" range="{-track} {track}" damping="0.02"/>
      <geom name="cart_body" type="box" size="{cart_half_x} {cart_half_y} {cart_half_z}" rgba="0.10 0.32 0.68 1" contype="0" conaffinity="0"/>
      <geom name="distance_sensor" type="box" pos="{cart_half_x + 0.014} -0.122 0.018" size="0.018 0.018 0.018" rgba="0.05 0.11 0.14 1" contype="0" conaffinity="0"/>
      <geom name="left_wheel" type="cylinder" pos="{-cart_half_x * 0.58} -0.135 {-cart_half_z}" size="0.038 0.018" euler="1.5707963268 0 0" rgba="0.03 0.04 0.05 1" contype="0" conaffinity="0"/>
      <geom name="right_wheel" type="cylinder" pos="{cart_half_x * 0.58} -0.135 {-cart_half_z}" size="0.038 0.018" euler="1.5707963268 0 0" rgba="0.03 0.04 0.05 1" contype="0" conaffinity="0"/>
    </body>
    <camera name="overview" pos="0 -4.2 1.45" xyaxes="1 0 0 0 0.34 0.94" fovy="38"/>
    <camera name="side" pos="0 -2.8 0.68" xyaxes="1 0 0 0 0.14 0.99" fovy="42"/>
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
