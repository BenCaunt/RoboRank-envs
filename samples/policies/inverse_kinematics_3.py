import math


class RobotPolicy:
    def step(self, arm):
        target = arm.get_target()
        spacing = arm.base_spacing_m
        l1 = arm.link_1_m
        l2 = arm.link_2_m

        left_anchor = (-spacing / 2.0, 0.0)
        right_anchor = (spacing / 2.0, 0.0)
        left_angle = _two_link_base_angle(left_anchor, (target.x, target.y), l1, l2, -1.0)
        right_angle = _two_link_base_angle(right_anchor, (target.x, target.y), l1, l2, 1.0)

        arm.submit_joint_angles([left_angle, right_angle])


def _two_link_base_angle(anchor, target, l1, l2, sign):
    dx = target[0] - anchor[0]
    dy = target[1] - anchor[1]
    distance = max(1e-9, math.hypot(dx, dy))
    base = math.atan2(dy, dx)
    cos_alpha = (l1 * l1 + distance * distance - l2 * l2) / (2.0 * l1 * distance)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)
    return base + sign * alpha
