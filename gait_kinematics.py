# %%
from __future__ import annotations

import csv
import json

from dataclasses import dataclass
from enum import Enum
from itertools import accumulate
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


def interweave(list1: list, list2: list) -> list:
    return [val for pair in zip(list1, list2) for val in pair]


# TODO: fix this mess
def save_data(data: list[list[list[Pose]]], filename: str):
    """Write pose data to the a CSV file."""

    with open(filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE)

        data = np.asarray(data)
        data = data.flatten()

        one_row = []
        for num in data:
            if len(one_row) < 32:
                one_row.append(num)
            else:
                writer.writerow(one_row)
                one_row = [num]


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


@dataclass(repr=False)
class Pose:
    """A pose in 2D space."""

    point: Point
    angle: float

    def __repr__(self):
        return f"(({self.point.x}, {self.point.y}), {self.angle})"


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

        # Convert from parent-relative to global angles
        self.angles = list(accumulate(angles))
        self.limits = limits
        self.lengths = lengths

        self.max_reach = sum(self.lengths)

        # Should be set after initial leg creation
        self.ground: float | None = None

    def global_joint_poses(self) -> list[Pose]:
        """Compute the global position and angle of each joint."""

        # Position and angle of hip
        poses = [Pose(self.hip_position, self.angles[0])]

        for i in range(self.num_segments):
            parent_angle = poses[-1].angle

            x = poses[-1].point.x + self.lengths[i] * cos(parent_angle)
            y = poses[-1].point.y + self.lengths[i] * sin(parent_angle)

            # No angle for tip of leg (foot)
            angle = self.angles[i + 1] if i < self.num_segments - 1 else inf

            poses.append(Pose(Point(x=x, y=y), angle))

        return poses

    # TODO: fix this mess
    def get_angles(self) -> list[float]:
        poses = self.global_joint_poses()
        angles = []
        for i, pose in enumerate(poses):
            if i + 1 >= len(poses):
                continue
            angles.append(pose.angle)
        return angles

    def foot_position(self) -> Point:
        return self.global_joint_poses()[-1].point

    def lowest_y(self) -> float:
        """Get the lowest point of the leg."""
        return min([pose.point.y for pose in self.global_joint_poses()])

    def raise_hip(self, y_offset: float = 0.1) -> None:
        """Raise the hip a bit to make it visible in the animation."""
        self.hip_position = Point(self.hip_position.x, self.hip_position.y + y_offset)

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

        if (goal - self.hip_position).norm() > self.max_reach:
            logger.warning(f"The position {goal} is beyond the reach of the leg.")

        if self.ground is None:
            raise ValueError("Ground level not set.")

        prev_distance_to_goal = inf

        # TODO: switch to relative angles?
        for _ in range(max_steps):

            # Adjustment so that no body parts dip below the ground plane
            if self.lowest_y() <= self.ground:

                delta = abs(self.lowest_y() - self.ground)

                base_rotation = -0.1
                rotation = base_rotation * rotation_factor

                for i in range(1, self.num_segments):

                    parent_angle = 0 if i == 0 else self.angles[i - 1]

                    lo_limit = self.limits[i][0] + parent_angle
                    hi_limit = self.limits[i][1] + parent_angle

                    joint_poses = self.global_joint_poses()

                    new_angle = wrap_to_pi(joint_poses[i].angle + rotation)

                    # Compute the new angle and clip within specified limits
                    # TODO: remove for now...
                    # self.angles[i] = clip(new_angle, lo_limit, hi_limit)

                    # TODO: unexplained
                    rotation *= -1

            # IK: start at the ankle and work to the hip (no joint at foot)
            for i in range(self.num_segments - 1, -1, -1):

                parent_angle = 0 if i == 0 else self.angles[i - 1]

                lo_limit = self.limits[i][0] + parent_angle
                hi_limit = self.limits[i][1] + parent_angle

                joint_poses = self.global_joint_poses()

                joint_to_foot = joint_poses[-1].point - joint_poses[i].point
                joint_to_goal = goal - joint_poses[i].point

                rotation_amount = Point.angle_between(joint_to_foot, joint_to_goal)

                # FIXME: Something is wrong here... rotation_amount is in degrees...
                # Just get rid of conversion?
                new_angle = wrap_to_pi(joint_poses[i].angle + rotation_amount)

                # Compute the new angle and clip within specified limits
                # TODO: remove for now...
                # self.angles[i] = clip(new_angle, lo_limit, hi_limit)

                self.angles[i] = new_angle

            # Check if close enough to goal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               l
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
        self.ground = self.legs[0].global_joint_poses()[-1].point.y
        for leg in self.legs:
            leg.ground = self.ground

    def __animate(
        # TODO: add type alias for this mess
        self,
        frames: list[list[tuple[list[float], list[float]]]],
    ) -> FuncAnimation:
        """Create animation of the animat walking."""

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

    def run_gait(
        self, job_dict: dict, kinematics_path: Path, animations_path: Path
    ) -> None:
        """Run the gait according to the gait dict."""

        horz_reach: float = job_dict["horizontal_reach"]
        vert_reach: float = job_dict["vertical_reach"]

        foot_order: list[list[int]] = job_dict["foot_order"]

        rota_factor: float = job_dict["rotation_factor"]

        # Initial position given by current leg positions should be a list[list[Point]]
        foot_positions = [[leg.foot_position()] for leg in self.legs]
        # xs = [pos[0].x for pos in positions]
        # ys = [pos[0].y for pos in positions]

        # Number of positions along the gait
        num_steps = 16
        x_delta = horz_reach / num_steps

        # Vertical motion is up then down
        y_delta = vert_reach / (num_steps // 2)

        # 0 = staging, 1 = forward, 2 = back, 3 = reposition, 4 = done
        foot_stages = [FootStage.PLANTED] * len(self.legs)

        num_foot_actions = len(foot_order)
        foot_action_index = 0

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
            # TODO: change to foot-index
            for i, leg in enumerate(self.legs):

                # No need to move, just copy the current position
                if (
                    foot_stages[i] == FootStage.PLANTED
                    or foot_stages[i] == FootStage.DONE
                ):
                    foot_positions[i] += [
                        foot_positions[i][-1] for _ in range(num_steps)
                    ]

                # Leg should move forward
                elif foot_stages[i] == FootStage.FORWARD:
                    new_x = foot_positions[i][-1].x
                    new_y = foot_positions[i][-1].y
                    for step in range(num_steps):
                        new_x += x_delta
                        new_y += y_delta if step < num_steps // 2 else -y_delta

                        foot_positions[i].append(Point(x=new_x, y=new_y))

                        # xs[i] += x_delta[i]
                        # ys[i] += y_delta[i] if step < num_steps // 2 else -y_delta[i]
                        # if step > num_steps // 2 and ys[i] < self.ground:
                        # foot_positions[i].append(Point(x=xs[i], y=self.ground))
                        # else:
                        # foot_positions[i].append(Point(x=xs[i], y=ys[i]))

                    foot_stages[i] = FootStage.BACKWARD

                # Leg should move backward
                elif foot_stages[i] == FootStage.BACKWARD:
                    # TODO: list comprehension?
                    new_x = foot_positions[i][-1].x
                    for step in range(int(num_steps * 1.5)):
                        new_x -= x_delta
                        foot_positions[i].append(Point(x=new_x, y=self.ground))

                    foot_stages[i] = FootStage.REPOSITION

                # Leg should reposition
                elif foot_stages[i] == FootStage.REPOSITION:
                    new_x = foot_positions[i][-1].x
                    new_y = foot_positions[i][-1].y
                    for step in range(num_steps // 2):
                        new_x += x_delta
                        new_y += y_delta if step < num_steps // 4 else -y_delta
                        foot_positions[i].append(
                            Point(
                                x=new_x, y=new_y if new_y > self.ground else self.ground
                            )
                        )
                        # xs[i] += x_delta[i]
                        # ys[i] += y_delta[i] if step < num_steps // 4 else -y_delta[i]
                        # if ys[i] < self.ground:  # clip it so it doesn't go below
                        # ys[i] = self.ground
                        # foot_positions[i].append(Point(x=xs[i], y=ys[i]))

                    foot_stages[i] = FootStage.DONE

                # Leg is done, reset it to planted
                elif foot_stages[i] == FootStage.DONE:
                    foot_stages[i] = FootStage.PLANTED

        initial_pts = self.get_pts_from_gjp()
        # for each leg, we're going to run gjp on it, strip only the points out, and then separate the points into tuples of x,y
        anim_frames = [
            [
                self.split_pts(initial_pts[0]),
                self.split_pts(initial_pts[1]),
                self.split_pts(initial_pts[2]),
                self.split_pts(initial_pts[3]),
            ]
        ]  # each frame contains info for one step of all 4 legs

        save_frames_angles = [
            [
                "FL A1 DF 1",
                "FL A1 DF 2",
                "FL A2 DF 1",
                "FL A2 DF 2",
                "FL A3 DF 1",
                "FL A3 DF 2",
                "FR A1 DF 1",
                "FR A1 DF 2",
                "FR A2 DF 1",
                "FR A2 DF 2",
                "FR A3 DF 1",
                "FR A3 DF 2",
                "BL A1 DF 1",
                "BL A1 DF 2",
                "BL A2 DF 1",
                "BL A2 DF 2",
                "BL A3 DF 1",
                "BL A3 DF 2",
                "BR A1 DF 1",
                "BR A1 DF 2",
                "BR A2 DF 1",
                "BR A2 DF 2",
                "BR A3 DF 1",
                "BR A3 DF 2",
                "SP A1 DF 1",
                "SP A1 DF 2",
                "SP A2 DF 1",
                "SP A2 DF 2",
                "T FL",
                "T FR",
                "T BL",
                "T BR",
            ]
        ]

        torso = [0.0, 0.0, 0.0, 0.0]  # two connections, 2 DOF each

        touch_sensors = [
            1.0 if (leg.lowest_y() <= self.ground) else 0.0 for leg in self.legs
        ]

        # TODO: get this section into a fx, since it gets re-used /*
        angle_frame = []
        touch_sensors = []

        for leg in self.legs:
            # frame.append(leg.global_joint_poses()[1])
            angles = leg.get_angles()
            angles = np.array(interweave(angles, np.zeros(len(angles)))).flatten()
            angle_frame = np.append(angle_frame, angles)

            if leg.lowest_y() <= self.ground:
                touch_sensors.append(1.0)
            else:
                touch_sensors.append(0.0)

        angle_frame = np.append(angle_frame, torso)
        angle_frame = np.append(angle_frame, touch_sensors)

        # save_frames.append(frame)
        save_frames_angles.append(angle_frame)
        # */

        # Compute joint angles for each point along the path
        # weird structure bc we want them separated by frames and not by leg
        # again, assuming all lens same
        for goal_idx in range(len(foot_positions[0])):

            for foot_index, leg in enumerate(self.legs):
                leg.move_foot(foot_positions[foot_index][goal_idx], rota_factor)

            # animate
            all_pts = self.get_pts_from_gjp()

            anim_frames.append(
                [
                    self.split_pts(all_pts[0]),
                    self.split_pts(all_pts[1]),
                    self.split_pts(all_pts[2]),
                    self.split_pts(all_pts[3]),
                ]
            )

            # save results
            # if we're not animating, we need the full poses
            # frame = []
            angle_frame = []
            touch_sensors = []
            torso = [0.0, 0.0, 0.0, 0.0]  # two connections, 2 DOF each

            for leg in self.legs:
                # frame.append(leg.global_joint_poses()[1])
                angles = leg.get_angles()
                angles = np.array(interweave(angles, np.zeros(len(angles)))).flatten()
                angle_frame = np.append(angle_frame, angles)

                if leg.lowest_y() <= self.ground + 0.05:
                    touch_sensors.append(1.0)
                else:
                    touch_sensors.append(0.0)

            angle_frame = np.append(angle_frame, torso)
            angle_frame = np.append(angle_frame, touch_sensors)

            # save_frames.append(frame)
            save_frames_angles = np.append(save_frames_angles, angle_frame)

        animation = self.__animate(anim_frames)
        HTML(animation.to_jshtml())

        gait_name = job_dict["name"]
        animation.save(str(animations_path / f"{gait_name}.gif"))
        save_data(
            save_frames_angles, str(kinematics_path / f"{gait_name}_kinematic.csv")
        )

    def split_pts(self, pts: list[Point]) -> tuple[list[float], list[float]]:
        # helper function, since we can update an actor with all x and y coordinates in this format
        xs = []
        ys = []
        for pt in pts:
            xs.append(pt.x)
            ys.append(pt.y)

        return (xs, ys)

    def get_pts_from_gjp(self) -> list[list[Point]]:
        # runs global_joint_poses on each leg in self.legs and returns a list of points for each leg
        all_pts = []

        for leg in self.legs:

            leg_pts = [pose.point for pose in leg.global_joint_poses()]
            all_pts.append(leg_pts)

        return all_pts


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
