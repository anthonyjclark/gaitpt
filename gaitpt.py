#%%
from __future__ import annotations
from dataclasses import dataclass
from loguru import logger
from math import atan2, cos, inf, pi, sin, sqrt, radians, degrees
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def clip(val: float, lo: float, hi: float) -> float:
    return max(min(val, hi), lo)


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0

    @property
    def norm(self) -> float:
        return sqrt(self.x ** 2 + self.y ** 2)

    def __add__(self, rhs) -> Point:
        return Point(x=self.x + rhs.x, y=self.y + rhs.y)

    def __sub__(self, rhs) -> Point:
        return Point(x=self.x - rhs.x, y=self.y - rhs.y)

    @classmethod
    def angle_between(cls, pt1: Point, pt2: Point) -> float:
        pt1_angle = atan2(pt1.y, pt1.x)
        pt2_angle = atan2(pt2.y, pt2.x)
        return (pt2_angle - pt1_angle) % (2.0 * pi)


@dataclass
class Pose:
    point: Point
    angle: float


class Animat:
    def __init__(self) -> None:
        # Towards Dynamic Trot Gait Locomotion—Design, Control, and Experiments with
        # Cheetah-cub, a Compliant Quadruped Robot
        # Alexander Spröwitz∗, Alexandre Tuleu, Massimo Vespignani, Mostafa Ajallooeian,
        # Emilie Badri, Auke Jan Ijspeert
        front_leg_angles = [radians(300), radians(220), radians(300)]
        front_leg_limits = [
            (radians(270), radians(340)),
            (radians(190), radians(360)),
            (radians(0), radians(170)),
        ]
        front_leg_lengths = [0.26, 0.42, 0.32]

        rear_leg_angles = front_leg_angles
        rear_leg_limits = front_leg_limits
        rear_leg_lengths = [0.41, 0.42, 0.17]

        self.rear_left = Leg(rear_leg_angles, rear_leg_limits, rear_leg_lengths)
        self.legs = [self.rear_left]

    def walk(self, animate=True):

        # TODO: figure out resting positions
        left_leg_positions = [self.rear_left.tip_position()]
        x, y = left_leg_positions[0].x, left_leg_positions[0].y
        delta_x = (1.0 - x) / 10

        # Forward motion
        for _ in range(10):
            x += delta_x
            left_leg_positions.append(Point(x=x, y=y))

        # Backward motion
        for _ in range(15):
            x -= delta_x
            left_leg_positions.append(Point(x=x, y=y))

        positions = [[p.point for p in self.rear_left.compute_joint_poses()]]

        for goal in left_leg_positions:
            self.rear_left.move_tip(goal)
            positions.append([p.point for p in self.rear_left.compute_joint_poses()])

        if animate:

            fig, ax = plt.subplots()

            lines = [ax.plot([], [], marker="o", linewidth=3)[0] for _ in range(3)]

            def init():
                ax.set_xlim([-4, 4])
                ax.set_ylim([-4, 4])
                return lines

            def update(frame, *fargs) -> Iterable:
                for i, actor in enumerate(lines):
                    base, tip = frame[i], frame[i + 1]
                    actor.set_data([base.x, tip.x], [base.y, tip.y])
                return lines

            animation = FuncAnimation(fig, update, frames=positions, init_func=init)
            return positions, animation

        return positions


class Leg:
    def __init__(
        self,
        angles: list[float],
        limits: list[tuple[float, float]],
        lengths: list[float],
    ) -> None:
        self.num_segments = 3
        assert len(angles) == self.num_segments
        assert len(limits) == self.num_segments
        assert len(lengths) == self.num_segments
        self.angles = angles.copy()  # Absolute angle
        self.limits = limits  # Relative to joint
        self.lengths = lengths
        self.max_reach = sum(self.lengths)

    def compute_joint_poses(self) -> list[Pose]:
        # Position and angle of hip
        poses = [Pose(Point(), self.angles[0])]

        for i in range(self.num_segments):
            x = poses[-1].point.x + self.lengths[i] * cos(self.angles[i])
            y = poses[-1].point.y + self.lengths[i] * sin(self.angles[i])
            a = poses[-1].angle + self.angles[i + 1] if i < self.num_segments - 1 else 0
            poses.append(Pose(Point(x=x, y=y), a))

        return poses

    def tip_position(self) -> Point:
        return self.compute_joint_poses()[-1].point

    def move_tip(
        self, goal: Point, max_steps: int = 100, tolerance: float = 1e-1
    ) -> None:
        """Move the tip using Cyclic Coordinate Descent Inverse Kinematics (CCD-IK).

        Args:
            goal (Point): desired position of the tip
            max_steps (int, optional): steps allowed to reach position. Defaults to 100.
            TODO: doc string
        """
        if goal.norm > self.max_reach:
            logger.warning(f"The position {goal} is beyond the reach of the leg.")

        prev_dist = inf

        for _ in range(max_steps):

            # Index zero of joint_poses is always (0, 0), and the final index
            # always gives the location of the tip (one more position than segment)
            for i in range(self.num_segments - 1, -1, -1):
                joint_poses = self.compute_joint_poses()
                joint_to_tip = joint_poses[-1].point - joint_poses[i].point
                joint_to_goal = goal - joint_poses[i].point

                rotation_angle = Point.angle_between(joint_to_tip, joint_to_goal)
                new_angle = self.angles[i] + rotation_angle
                parent_angle = self.angles[i - 1] if i > 0 else 0
                lo = self.limits[i][0] + parent_angle
                hi = self.limits[i][1] + parent_angle
                print(
                    i,
                    f"{degrees(parent_angle):.3f}",
                    f"{degrees(self.angles[i]):.3f}",
                    f"{degrees(rotation_angle):.3f}",
                    f"{degrees(new_angle):.3f}",
                    f"{degrees(lo):.3f}",
                    f"{degrees(hi):.3f}",
                )
                self.angles[i] = new_angle % (2.0 * pi)
                # clip(new_angle, lo, hi) % (2.0 * pi)

            # Check if close enough to goal
            dist = (goal - self.tip_position()).norm
            if abs(dist) < tolerance:
                break

            # Check if still making progress (goal might be out of reach)
            if abs(dist - prev_dist) < tolerance:
                break

            prev_dist = dist


animat = Animat()
_, animation = animat.walk()
HTML(animation.to_jshtml())


# %%
