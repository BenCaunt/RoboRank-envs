import math


class RobotPolicy:
    def step(self, arm):
        target = arm.get_target()
        yaw = math.atan2(target.y, target.x)
        radial = math.hypot(target.x, target.y)
        vertical = target.z - arm.base_height_m
        l1 = arm.link_1_m
        l2 = arm.link_2_m

        cos_elbow = (radial * radial + vertical * vertical - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        candidates = []
        for elbow in (math.acos(cos_elbow), -math.acos(cos_elbow)):
            shoulder = math.atan2(vertical, radial) - math.atan2(
                l2 * math.sin(elbow),
                l1 + l2 * math.cos(elbow),
            )
            candidates.append((shoulder, elbow))

        shoulder_limits = arm.joint_limits_rad[1]
        elbow_limits = arm.joint_limits_rad[2]
        shoulder, elbow = min(
            candidates,
            key=lambda item: _limit_error(item[0], shoulder_limits) + _limit_error(item[1], elbow_limits),
        )

        arm.submit_joint_angles([yaw, shoulder, elbow])


def _limit_error(value, limits):
    lower, upper = limits
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0
