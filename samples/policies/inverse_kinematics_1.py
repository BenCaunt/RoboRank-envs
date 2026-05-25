import math


class RobotPolicy:
    def step(self, arm):
        target = arm.get_target()
        x = target.x
        y = target.y
        l1 = arm.link_1_m
        l2 = arm.link_2_m

        cos_elbow = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow = math.acos(cos_elbow)
        shoulder = math.atan2(y, x) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))

        arm.submit_joint_angles([shoulder, elbow])
