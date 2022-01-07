#%%
from __future__ import annotations
from dataclasses import dataclass
from itertools import accumulate
from loguru import logger
from math import atan2, cos, inf, pi, sin, sqrt, radians, degrees
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def clip(val: float, lo: float, hi: float) -> float:
    return max(min(val, hi), lo)


def deg(deg: float) -> float:
    return rad(radians(deg))


def rad(rad: float) -> float:
    rad = rad % (2 * pi)
    return rad if rad < pi else rad - 2 * pi


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
        return rad(pt2_angle - pt1_angle)


@dataclass
class Pose:
    point: Point
    angle: float


class Animat:
    def __init__(self) -> None:
        """Implementing a four-legged, walking animat.

        Some dimensions and angles inspired by:
            Towards Dynamic Trot Gait Locomotion—Design, Control, and Experiments with
            Cheetah-cub, a Compliant Quadruped Robot
            by Alexander Spröwitz, Alexandre Tuleu, Massimo Vespignani, Mostafa Ajallooeian,
            Emilie Badri, Auke Jan Ijspeert
        """

        # Angle and limits are relative to parent joint (or the world in the case of
        # the firt segment)
        front_leg_angles = [deg(300), deg(-80), deg(80)]
        front_leg_limits = [
            (deg(270), deg(340)),
            (deg(-160), deg(20)),
            (deg(20), deg(160)),
        ]
        front_leg_lengths = [0.26, 0.42, 0.32]

        rear_leg_angles = front_leg_angles
        rear_leg_limits = front_leg_limits
        rear_leg_lengths = [0.41, 0.42, 0.17]

        self.rear_left = Leg(rear_leg_angles, rear_leg_limits, rear_leg_lengths)
        self.legs = [self.rear_left]

    def _animate(self, positions: list[list[Point]]) -> FuncAnimation:

        # TODO: should work with more than one leg (list[list[list[Point]]])

        fig, ax = plt.subplots()

        lines = [ax.plot([], [], marker="o", linewidth=3)[0] for _ in range(3)]
        ax.plot([-1, 1], [positions[0][-1].y] * 2, "k")

        def init():
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            return lines

        def update(frame, *fargs) -> Iterable:
            for i, actor in enumerate(lines):
                base, tip = frame[i], frame[i + 1]
                actor.set_data([base.x, tip.x], [base.y, tip.y])
            return lines

        animation = FuncAnimation(fig, update, frames=positions, init_func=init)
        return animation

    def walk(self, animate=True) -> list[list[Point]] | FuncAnimation:
        """Compute joint angles for a walking gait.

        Args:
            animate (bool, optional): return an animation. Defaults to True.

        Returns:
            list[list[Pose]] | FuncAnimation: pose data or animation
        """

        # TODO: should work with more than one leg (list[list[list[Point]]])

        horiz_reach = 0.6
        vertical_reach = 0.4

        # Initial position given by current leg position
        initial_position = self.rear_left.tip_position()

        # Forward motion path
        num_steps = 16

        x, y = initial_position.x, initial_position.y
        delta_x = horiz_reach / num_steps
        delta_y = 2 * vertical_reach / num_steps

        left_leg_positions = [initial_position]

        for step in range(num_steps):
            x += delta_x
            y += delta_y if step < num_steps // 2 else -delta_y
            left_leg_positions.append(Point(x=x, y=y))

        # Backward motion path
        for _ in range(int(num_steps * 1.5)):
            x -= delta_x
            left_leg_positions.append(Point(x=x, y=y))

        # Path back to initial position
        for step in range(num_steps // 2):
            x += delta_x
            y += delta_y if step < num_steps // 4 else -delta_y
            left_leg_positions.append(Point(x=x, y=y))

        positions = [[p.point for p in self.rear_left.global_joint_poses()]]

        # Compute joint angles for each point along th path
        for goal in left_leg_positions:
            self.rear_left.move_tip(goal)
            positions.append([p.point for p in self.rear_left.global_joint_poses()])

        return self._animate(positions) if animate else positions


class Leg:
    def __init__(
        self,
        angles: list[float],
        limits: list[tuple[float, float]],
        lengths: list[float],
    ) -> None:
        """Create a single leg.

        Args:
            angles (list[float]): list of angles relative to their preceding angle
            limits (list[tuple[float, float]]): list of angle limits relative to parent joint
            lengths (list[float]): list of leg lengths
        """

        self.num_segments = 3

        assert len(angles) == self.num_segments
        assert len(limits) == self.num_segments
        assert len(lengths) == self.num_segments

        self.angles = list(accumulate(angles))  # Convert from relative to global
        self.limits = limits
        self.lengths = lengths

        self.max_reach = sum(self.lengths)

    def global_joint_poses(self) -> list[Pose]:
        """Compute the global position and angle of each joint.

        Returns:
            list[Pose]: global position and angle of each joint
        """

        # Position and angle of hip
        poses = [Pose(Point(), self.angles[0])]

        for i in range(self.num_segments):
            parent_angle = poses[-1].angle

            x = poses[-1].point.x + self.lengths[i] * cos(parent_angle)
            y = poses[-1].point.y + self.lengths[i] * sin(parent_angle)

            angle = self.angles[i + 1] if i < self.num_segments - 1 else inf

            poses.append(Pose(Point(x=x, y=y), angle))

        return poses

    def tip_position(self) -> Point:
        """Compute the position of the tip/foot.

        Returns:
            Point: position of the tip/foot
        """
        return self.global_joint_poses()[-1].point

    def move_tip(
        self, goal: Point, max_steps: int = 100, tolerance: float = 1e-1
    ) -> None:
        """Move the foot/tip using Cyclic Coordinate Descent Inverse Kinematics (CCD-IK).

        Args:
            goal (Point): goal position
            max_steps (int, optional): maximum steps to use. Defaults to 100.
            tolerance (float, optional): required closeness. Defaults to 1e-1.
        """

        if goal.norm > self.max_reach:
            logger.warning(f"The position {goal} is beyond the reach of the leg.")

        prev_dist = inf

        for _ in range(max_steps):

            # len(joint_poses) == 4 (three segments + tip)
            # joint_poses[0] is the base
            # joint_poses[-1] is the tip (no angle associated with tip)
            for i in range(self.num_segments - 1, -1, -1):

                joint_poses = self.global_joint_poses()

                joint_to_tip = joint_poses[-1].point - joint_poses[i].point
                joint_to_goal = goal - joint_poses[i].point

                rotation_amount = Point.angle_between(joint_to_tip, joint_to_goal)

                new_angle = rad(joint_poses[i].angle + rotation_amount)

                parent_angle = 0 if i == 0 else self.angles[i - 1]

                lo_limit = self.limits[i][0] + parent_angle
                hi_limit = self.limits[i][1] + parent_angle

                # Compute the new angle and clip within specified limits
                self.angles[i] = clip(new_angle, lo_limit, hi_limit)

            # Check if close enough to goal
            dist = (goal - self.tip_position()).norm
            if abs(dist) < tolerance:
                break

            # Check if still making progress (goal might be out of reach)
            if abs(dist - prev_dist) < tolerance:
                break

            prev_dist = dist


animat = Animat()
animation = animat.walk()
HTML(animation.to_jshtml())
animation.save("example.gif")


# %%
