from __future__ import annotations

import math


class RobotPolicy:
    def step(self, cart_pole: CartPole) -> None:
        state = cart_pole.get_state()
        minimum_phase = cart_pole.minimum_phase_output()
        gain = cart_pole.minimum_phase_gain_m

        # x = y - c sin(theta), so this feedback is organized around the
        # minimum-phase output while retaining direct damping on the pole states.
        force = (
            1.0 * minimum_phase.output_m
            + 2.0 * minimum_phase.output_velocity_mps
            + (25.0 - gain) * state.pole_angle_rad
            + (4.0 - 2.0 * gain * math.cos(state.pole_angle_rad)) * state.pole_angular_velocity_radps
        )
        cart_pole.set_force(_clamp(force, -cart_pole.max_force_n, cart_pole.max_force_n))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
