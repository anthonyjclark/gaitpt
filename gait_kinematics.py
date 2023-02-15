# %%
from __future__ import annotations

import csv
import json

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate, product
from math import atan2, cos, inf, pi, radians, sin, sqrt
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from loguru import logger
from matplotlib.animation import FuncAnimation

from pathlib import Path


def wrap_to_pi(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def deg2rad(angle: float) -> float:
    return wrap_to_pi(radians(angle))


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


# def interweave(list1: list, list2: list) -> list:
#     return [val for pair in zip(list1, list2) for val in pair]


def points_to_xy(points: list[Point]) -> tuple[list[float], list[float]]:
    """Convert a list of points to a list of line segments."""
    x = [point.x for point in points]
    y = [point.y for point in points]
    return x, y


class FootStage(Enum):
    PLANTED = 0
    FORWARD = 1
    BACKWARD = 2
    REPOSITION = 3
    DONE = 4


@dataclass
class Point:
    """A point in 2D space."""

    x: float = 0.0
    y: float = 0.0

    def norm(self) -> float:
        return sqrt(self.x**2 + self.y**2)

    def __add__(self, rhs) -> Point:
        return Point(x=self.x + rhs.x, y=self.y + rhs.y)

    def __sub__(self, rhs) -> Point:
        return Point(x=self.x - rhs.x, y=self.y - rhs.y)

    @classmethod
    def angle_between(cls, pt1: Point, pt2: Point) -> float:
        pt1_angle = atan2(pt1.y, pt1.x)
        pt2_angle = atan2(pt2.y, pt2.x)
        return wrap_to_pi(pt2_angle - pt1_angle)


class Leg:
    """A single leg of the animat."""

    def __init__(
        self,
        angles: list[float],
        limits: list[tuple[float, float]],
        lengths: list[float],
        hip_position: Point = Point(),
    ) -> None:
        """Create a single leg.

        Args:
            angles (list[float]): list of angles, each relative to preceding parent joint
            limits (list[tuple[float, float]]): list of (lo, hi) joint limits
            lengths (list[float]): list of leg lengths
            hip (Point, optional): location of the hip Defaults to Point().
        """

        self.num_segments = 3
        self.hip_position = hip_position

        assert len(angles) == self.num_segments
        assert len(limits) == self.num_segments
        assert len(lengths) == self.num_segments

        # self.relative_centers = deepcopy(angles)
        # self.relative_offsets = [0.0] * self.num_segments

        self.angles = list(accumulate(angles))
        self.limits = limits
        self.lengths = lengths

        self.reach = sum(self.lengths)

        # Should be set after initial leg creation
        self.ground: float | None = None

    def joint_points(self) -> list[Point]:
        points = [self.hip_position]
        for i in range(self.num_segments):
            x = points[-1].x + self.lengths[i] * cos(self.angles[i])
            y = points[-1].y + self.lengths[i] * sin(self.angles[i])
            points.append(Point(x=x, y=y))

        return points

    def foot_position(self) -> Point:
        return deepcopy(self.joint_points()[-1])

    def foot_y(self) -> float:
        return self.foot_position().y

    def move_foot(
        self,
        goal: Point,
        rotation_factor: float,
        max_steps: int = 100,
        tolerance: float = 1e-1,
    ) -> None:
        """Move the foot/tip using Cyclic Coordinate Descent Inverse Kinematics (CCD-IK).

        Args:
            goal (Point): goal position
            rotation_factor (float): how much to rotate the leg
            max_steps (int, optional): maximum steps to use. Defaults to 100.
            tolerance (float, optional): required closeness. Defaults to 1e-1.

        Raises:
            ValueError: if ground level not set
        """

        if (goal - self.hip_position).norm() > self.reach:
            logger.warning(f"The position {goal} is beyond the reach of the leg.")

        if self.ground is None:
            raise ValueError("Ground level not set.")

        prev_distance_to_goal = inf

        for _ in range(max_steps):

            # TODO: do this with better initial conditions?
            # # Adjustment so that no body parts dip below the ground plane
            # if self.lowest_y() <= self.ground:

            #     delta = abs(self.lowest_y() - self.ground)

            #     base_rotation = -0.1
            #     rotation = base_rotation * rotation_factor

            #     for i in range(1, self.num_segments):

            #         parent_angle = 0 if i == 0 else self.angles[i - 1]

            #         lo_limit = self.limits[i][0] + parent_angle
            #         hi_limit = self.limits[i][1] + parent_angle

            #         joint_poses = self.global_joint_poses()

            #         new_angle = wrap_to_pi(joint_poses[i].angle + rotation)

            #         # Compute the new angle and clip within specified limits
            #         # TODO: remove for now...
            #         # self.angles[i] = clip(new_angle, lo_limit, hi_limit)

            #         # TODO: unexplained
            #         rotation *= -1

            # IK: start at the ankle and work to the hip (no joint at foot)
            for j in range(self.num_segments - 1, -1, -1):

                parent_joint_angle = 0 if j == 0 else self.angles[j - 1]
                joint_lo = self.limits[j][0] + parent_joint_angle
                joint_hi = self.limits[j][1] + parent_joint_angle

                joint_points = self.joint_points()
                joint_to_foot = joint_points[-1] - joint_points[j]
                joint_to_goal = goal - joint_points[j]

                # New joint angle will place the foot as close as possible to the goal
                displacement_to_goal = Point.angle_between(joint_to_foot, joint_to_goal)
                new_angle = wrap_to_pi(self.angles[j] + displacement_to_goal)

                # Compute the new angle and clip within specified limits
                self.angles[j] = wrap_to_pi(clip(new_angle, joint_lo, joint_hi))

            # Check if close enough to goal
            foot_distance_to_goal = abs((goal - self.foot_position()).norm())
            if foot_distance_to_goal < tolerance:
                break

            # Check if still making progress (goal might be out of reach)
            distance_since_last_step = abs(
                foot_distance_to_goal - prev_distance_to_goal
            )
            if distance_since_last_step < tolerance:
                break

            prev_distance_to_goal = foot_distance_to_goal


class QuadrupedAnimat:
    """Kinematics for a four-legged, walking animat.

    Some dimensions and angles inspired by:
        Towards Dynamic Trot Gait Locomotion—Design, Control, and Experiments with
        Cheetah-cub, a Compliant Quadruped Robot
        by Alexander Spröwitz, Alexandre Tuleu, Massimo Vespignani, Mostafa Ajallooeian,
        Emilie Badri, Auke Jan Ijspeert
    """

    def __init__(self, file: str) -> None:

        with open(file, "r") as json_file:
            animat = json.load(json_file)["animat"]

        front_hip_pos = Point(animat["front_hip_x"], animat["hip_y"])
        back_hip_pos = Point(animat["back_hip_x"], animat["hip_y"])

        self.legs: list[Leg] = []

        for leg_dict in animat["legs"]:

            angles = [deg2rad(angle) for angle in leg_dict["angles"]]
            limits = [(deg2rad(lm[0]), deg2rad(lm[1])) for lm in leg_dict["limits"]]
            lengths = leg_dict["lengths"]
            hip_pos = front_hip_pos if leg_dict["hip"] == "front" else back_hip_pos

            self.legs.append(Leg(angles, limits, lengths, hip_pos))
            self.legs.reverse()

        # Location of ground below "hip_y" is unknown until we have posed the legs
        self.ground = self.legs[0].joint_points()[-1].y
        for leg in self.legs:
            leg.ground = self.ground

    def __animate(
        self,
        frames: list[list[tuple[list[float], list[float]]]],
    ) -> FuncAnimation:
        """Generate an animation given the list of line segment frames.

        Args:
            frames (list[list[tuple[list[float], list[float]]]]): frames->legs->(xs, ys)

        Returns:
            FuncAnimation: matplotlib animation object
        """

        fig, ax = plt.subplots()

        # One for each leg. left side legs marked with circle, right with square
        lines = [
            ax.plot([], [], marker="o", linewidth=3)[0],
            ax.plot([], [], marker="s", linewidth=3)[0],
            ax.plot([], [], marker="o", linewidth=3)[0],
            ax.plot([], [], marker="s", linewidth=3)[0],
        ]

        # Ground - first frame, second leg, rt side of tuple (y), last one (tip)
        ax.plot([-1, 1], [self.ground] * 2, "k")

        def init():
            """Figure axis setup."""
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            return lines

        def update(frame, *fargs) -> Iterable:
            """Figure frame update."""
            # One frame would have one list of legs, each with an x,y list tuple
            for i, actor in enumerate(reversed(lines)):
                actor.set_data(frame[i][0], frame[i][1])
            return lines

        animation = FuncAnimation(fig, update, frames=frames, init_func=init)
        return animation

    def leg_points(self) -> list[list[Point]]:
        """Return the points of each leg."""
        return [leg.joint_points() for leg in self.legs]

    def run_gait(
        self, gait_config: dict, kinematics_path: Path, animations_path: Path
    ) -> None:
        """Run the gait according to the gait dict."""

        horz_reach: float = gait_config["horizontal_reach"]
        vert_reach: float = gait_config["vertical_reach"]

        foot_order: list[list[int]] = gait_config["foot_order"]

        rota_factor: float = gait_config["rotation_factor"]

        # Initial position given by initial leg positions should be a list[list[Point]]
        foot_positions = [[leg.foot_position()] for leg in self.legs]

        # Number of positions along the gait
        num_steps = 16
        x_delta = horz_reach / num_steps

        # Vertical motion is up then down
        y_delta = vert_reach / (num_steps // 2)

        # All feet start in the planted stage
        foot_stages = [FootStage.PLANTED] * len(self.legs)

        num_foot_actions = len(foot_order)
        foot_action_index = 0

        # Manually compute the path of each foot (no kinematics yet)
        while any(stage != FootStage.DONE for stage in foot_stages):

            if foot_action_index < num_foot_actions:

                foot_action = foot_order[foot_action_index]

                # No need to change stage for this action
                if foot_action == 0:
                    foot_action_index += 1
                    continue

                # Start the specified legs moving forward
                for foot_index in foot_action:
                    foot_stages[foot_index] = FootStage.FORWARD

                foot_action_index += 1

            # Add new positions for each leg
            for i in range(len(self.legs)):

                foot_stage = foot_stages[i]
                foot_pos = deepcopy(foot_positions[i][-1])

                # No need to move, just copy the current position
                if foot_stage == FootStage.PLANTED or foot_stage == FootStage.DONE:
                    foot_positions[i] += [foot_pos for _ in range(num_steps)]

                # Leg should move forward
                elif foot_stage == FootStage.FORWARD:
                    for step in range(num_steps):
                        foot_pos.x += x_delta
                        foot_pos.y += y_delta if step < num_steps // 2 else -y_delta
                        foot_positions[i].append(Point(x=foot_pos.x, y=foot_pos.y))

                    foot_stages[i] = FootStage.BACKWARD

                # Leg should move backward on the ground
                elif foot_stage == FootStage.BACKWARD:
                    # Move back to neutral, and then back again halfway
                    for step in range(int(num_steps * 1.5)):
                        foot_pos.x -= x_delta
                        foot_positions[i].append(Point(x=foot_pos.x, y=self.ground))

                    foot_stages[i] = FootStage.REPOSITION

                # Leg should reposition
                elif foot_stage == FootStage.REPOSITION:
                    for step in range(num_steps // 2):
                        foot_pos.x += x_delta
                        foot_pos.y += y_delta if step < num_steps // 4 else -y_delta

                        new_y = foot_pos.y if foot_pos.y > self.ground else self.ground
                        foot_positions[i].append(Point(x=foot_pos.x, y=new_y))

                    foot_stages[i] = FootStage.DONE

                # Leg is done, reset it to planted
                elif foot_stage == FootStage.DONE:
                    foot_stages[i] = FootStage.PLANTED

        # Joint angles for each foot position
        # TODO: initial angles
        angle_data = []

        # Animation is created using line segments
        # For each time step, for each leg, for each x and y
        anim_data = [[points_to_xy(points) for points in self.leg_points()]]

        # Run IK for each foot position
        num_foot_positions = len(foot_positions[0])
        for pos_index in range(num_foot_positions):

            # Update the legs
            for foot_index, leg in enumerate(self.legs):
                leg.move_foot(foot_positions[foot_index][pos_index], rota_factor)

            # Get current angles for the CSV file
            # angle_data.append([leg.get_angles() for leg in self.legs])

            # Get current leg points for the animation
            anim_data.append([points_to_xy(points) for points in self.leg_points()])

        gait_name = gait_config["name"]

        animation = self.__animate(anim_data)
        animation.save(str(animations_path / f"{gait_name}.gif"))
        # HTML(animation.to_jshtml())

        # Order of joints in the CSV file (corresponds to simulation)
        hip_knee_ankle = "HKA"
        front_rear = "FR"
        left_right = "LR"
        dof = "12"

        # FLH_1, FLH_2, etc.
        csv_header = [
            f"{fr}{lr}{hka}_{dof}"
            for hka, fr, lr, dof in product(hip_knee_ankle, front_rear, left_right, dof)
        ]

        # Add touch sensors
        csv_header += ["FL_Touch", "FR_Touch", "RL_Touch", "RR_Touch"]

        # save_data(csv_data, str(kinematics_path / f"{gait_name}_kinematic.csv"))


if __name__ == "__main__":

    kinematics_path = Path("MotionData/")
    animations_path = Path("Animations/")

    animat_config_file = "dog_config.json"
    animat = QuadrupedAnimat(file=animat_config_file)

    with open(animat_config_file, "r") as config_file:
        config_file = json.load(config_file)
        gaits = config_file["gaits"]

    for gait in gaits:
        print("Running gait:", gait["name"])
        animat.run_gait(gait, kinematics_path, animations_path)
