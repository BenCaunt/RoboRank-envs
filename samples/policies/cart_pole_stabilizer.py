from __future__ import annotations


class RobotPolicy:
    def step(self, cart_pole: CartPole) -> None:
        state = cart_pole.get_state()
        x_error = state.cart_position_m - cart_pole.target_position_m
        force = (
            1.0 * x_error
            + 2.0 * state.cart_velocity_mps
            + 25.0 * state.pole_angle_rad
            + 4.0 * state.pole_angular_velocity_radps
        )
        cart_pole.set_force(_clamp(force, -cart_pole.max_force_n, cart_pole.max_force_n))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
