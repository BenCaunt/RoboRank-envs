class RobotPolicy:
    def step(self, cart):
        state = cart.get_state()
        target = cart.get_target()

        error = target.position_m - state.position_m
        direction = 1.0 if error >= 0.0 else -1.0
        distance = abs(error)
        velocity_along_path = state.velocity_mps * direction
        target_velocity = target.velocity_mps * direction
        max_accel = cart.max_acceleration_mps2
        max_velocity = cart.max_velocity_mps

        stopping_distance = max(0.0, velocity_along_path * velocity_along_path - target_velocity * target_velocity)
        stopping_distance /= 2.0 * max_accel

        if distance <= 0.01 and abs(state.velocity_mps - target.velocity_mps) <= 0.03:
            acceleration = 0.0
        elif velocity_along_path < -0.02:
            acceleration = direction * max_accel
        elif stopping_distance >= distance:
            acceleration = -direction * max_accel
        elif velocity_along_path < max_velocity - max_accel * cart.dt:
            acceleration = direction * max_accel
        else:
            acceleration = 0.0

        cart.set_acceleration(acceleration)
