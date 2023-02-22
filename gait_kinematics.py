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

# from IPython.display import HTML
from loguru import logger
from matplotlib.animation import FuncAnimation

from pathlib import Path


# TODO: remove
import sys
from math import degrees


def join_floats(vals: Iterable[float]) -> str:
    return ",".join([f"{val:.3f}" for val in vals])


def wrap_to_pi(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def deg2rad(angle: float) -> float:
    return wrap_to_pi(radians(angle))


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def interleave_with(vals: list[float], val: float) -> list[float]:
    return [val for pair in zip([val] * len(vals), vals) for val in pair]


def points_to_xy(points: list[Point]) -> tuple[list[float], list[float]]:
    """Convert a list of points to a list of line segments."""
    x = [point.x for point in points]
    y = [point.y for point in points]
    return x, y


class LegStage(Enum):
    PLANTED = 0
    LIFT = 1
    SUPPORT = 2
    THRUST = 3
    SWING = 4
    STEP = 5


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

        self.centers = deepcopy(angles)
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

    def foot_is_touching_ground(self, epsilon=0.05) -> float:
        """Return 1.0 if foot is touching ground, 0.0 otherwise."""
        ground_level = self.ground + epsilon if self.ground else 0.0
        return 1.0 if self.foot_position().y <= ground_level else 0.0

    def angle_offsets(self) -> list[float]:
        #  Hip: angle - (  0    + center)
        # !Hip: angle - (parent + center)
        return [
            wrap_to_pi(angle - (parent + center))
            for angle, parent, center in zip(
                self.angles, [0] + self.angles, self.centers
            )
        ]

    def move_foot(
        self,
        goal: Point,
        max_steps: int = 100,
        tolerance: float = 1e-1,
    ) -> None:
        """Move the foot/tip using Cyclic Coordinate Descent Inverse Kinematics (CCD-IK).

        Args:
            goal (Point): goal position
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

        front_hip_pos = Point(0.0, 0.0)
        rear_hip_pos = Point(-animat["length"], 0.0)

        self.legs: dict[str, Leg] = {}

        for leg_name, leg_dict in animat["legs"].items():

            angles = [deg2rad(angle) for angle in leg_dict["angles"]]
            limits = [(deg2rad(lm[0]), deg2rad(lm[1])) for lm in leg_dict["limits"]]
            lengths = leg_dict["lengths"]
            hip_pos = front_hip_pos if leg_name.startswith("front") else rear_hip_pos

            self.legs[leg_name] = Leg(angles, limits, lengths, hip_pos)

        # Location of ground below "hip_y" is unknown until we have posed the legs
        self.ground = self.legs["front_left"].joint_points()[-1].y
        for leg_name in self.legs:
            self.legs[leg_name].ground = self.ground

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

        # TODO: downsample frames https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html

        fig, ax = plt.subplots()

        # One for each leg. left side legs marked with circle, right with square
        lines = [
            ax.plot([], [], marker="o", linewidth=3)[0],
            ax.plot([], [], marker="s", linewidth=3)[0],
            ax.plot([], [], marker="o", linewidth=3)[0],
            ax.plot([], [], marker="s", linewidth=3)[0],
        ]

        # Add text for the time step
        step = 0
        text = ax.text(0, 1, f"Step {step} of {len(frames)}", transform=ax.transAxes)

        # Ground - first frame, second leg, rt side of tuple (y), last one (tip)
        ax.plot([-1, 1], [self.ground] * 2, "k")

        def init():
            """Figure axis setup."""
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            return lines

        def update(frame, *fargs) -> Iterable:
            """Figure frame update."""

            nonlocal step
            text.set_text(f"Step {step} of {len(frames)}")
            step += 1

            # Each frame has a list of points for each leg
            for i, actor in enumerate(lines):
                actor.set_data(frame[i][0], frame[i][1])
            return lines

        animation = FuncAnimation(fig, update, frames=frames, init_func=init)
        return animation

    def leg_points(self, order: list[str]) -> list[list[Point]]:
        """Return the points of each leg."""
        return [self.legs[leg_name].joint_points() for leg_name in order]

    def leg_angle_offsets(self, order: list[str]) -> list[list[float]]:
        """Return the angles of each leg relative to parent joints."""
        return [self.legs[leg_name].angle_offsets() for leg_name in order]

    def foot_touches(self, order: list[str]) -> list[float]:
        """Return foot touches for each leg as a float."""
        return [self.legs[leg_name].foot_is_touching_ground() for leg_name in order]

    def run_gait(
        self,
        gait_config: dict,
        kinematics_path: Path,
        animations_path: Path,
        num_cycles: int,
        num_steps: int,
    ) -> None:
        """Run the gait according to the gait dict."""

        horz_reach: float = gait_config["horizontal_reach"]
        vert_reach: float = gait_config["vertical_reach"]
        gait_startup: dict[str, list[str]] = gait_config["gait_startup"]
        gait_cycle: dict[str, list[str]] = gait_config["gait_cycle"]

        # Amount to move in horizontal direction each step
        x_delta = horz_reach / 2 / num_steps

        # Vertical motion is up then down
        y_delta = vert_reach / num_steps

        # All feet start in the standing stage
        leg_stages = {
            leg: gait_startup[leg] + gait_cycle[leg] * num_cycles
            for leg in gait_startup
        }

        # Initial position given by initial leg positions; should be a list[list[Point]]
        foot_positions = {leg: [self.legs[leg].foot_position()] for leg in self.legs}

        # Manually compute the path of each foot (no kinematics yet)
        num_stage_changes = len(leg_stages["front_left"])
        for stage_index in range(num_stage_changes):

            # Add new positions for each leg
            for leg_name in leg_stages:

                # Convert string to FootStage enum
                leg_stage = LegStage[leg_stages[leg_name][stage_index]]
                foot_pos = deepcopy(foot_positions[leg_name][-1])

                # New positions of the foot for this leg for this stage
                new_positions = []

                # TODO: these all do the same thing, just set a delta in the cases
                match leg_stage:
                    # No movement
                    case LegStage.PLANTED:
                        new_positions = [foot_pos for _ in range(num_steps)]

                    # Lift straight up
                    case LegStage.LIFT:
                        for _ in range(num_steps):
                            foot_pos.y += y_delta
                            new_positions.append(Point(x=foot_pos.x, y=foot_pos.y))

                    # Move from forward to middle on the ground
                    case LegStage.SUPPORT:
                        for _ in range(num_steps):
                            foot_pos.x -= x_delta
                            new_positions.append(Point(x=foot_pos.x, y=self.ground))

                    # Move from middle to back on the ground
                    case LegStage.THRUST:
                        for _ in range(num_steps):
                            foot_pos.x -= x_delta
                            new_positions.append(Point(x=foot_pos.x, y=self.ground))

                    # Move from back to middle in the air (going up)
                    case LegStage.SWING:
                        for _ in range(num_steps):
                            foot_pos.x += x_delta
                            foot_pos.y += y_delta
                            new_y = max(foot_pos.y, self.ground)
                            new_positions.append(Point(x=foot_pos.x, y=new_y))

                    # Move from middle to forward in the air (going down)
                    case LegStage.STEP:
                        for _ in range(num_steps):
                            foot_pos.x += x_delta
                            foot_pos.y -= y_delta
                            new_y = max(foot_pos.y, self.ground)
                            new_positions.append(Point(x=foot_pos.x, y=new_y))

                foot_positions[leg_name] += new_positions

        # Leg order in simulation
        leg_order = ["front_left", "front_right", "rear_left", "rear_right"]

        # Joint angles and touch sensors for each foot position
        angle_data = [self.leg_angle_offsets(leg_order)]
        touch_data = [self.foot_touches(leg_order)]

        # Animation is created using line segments
        # For each time step, for each leg, for each x and y
        anim_data = [[points_to_xy(points) for points in self.leg_points(leg_order)]]

        # Run IK for each foot position
        num_foot_positions = len(foot_positions["front_left"])
        for pos_index in range(num_foot_positions):

            # Update the legs
            for leg_name, leg in self.legs.items():
                leg.move_foot(foot_positions[leg_name][pos_index])

            # Get current angles for the CSV file
            angle_data.append(self.leg_angle_offsets(leg_order))
            touch_data.append(self.foot_touches(leg_order))

            # Get current leg points for the animation
            anim_data.append([points_to_xy(pts) for pts in self.leg_points(leg_order)])

            # print(
            #     angle_data[-1][0][0],
            #     self.legs["front_left"].angles[0],
            #     self.legs["front_left"].centers[0],
            #     self.legs["front_left"].centers[0] - self.legs["front_left"].angles[0],
            # )

            # absolute = list(map(degrees, self.legs["front_left"].angles))
            # relative = list(map(degrees, self.legs["front_left"].angle_offsets()))
            # footy = self.legs["front_left"].foot_position().y
            # print(join_floats(absolute + relative + [footy]), file=sys.stderr)

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

        csv_filename = f"{gait_name}_kinematic_new.csv"
        with open(csv_filename, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_header)

            for angles, touches in zip(angle_data, touch_data):

                row: list[float] = []

                # Write joint angles and take into account the simulation order and DOFs
                hips = [leg_angles[0] for leg_angles in angles]
                knees = [leg_angles[1] for leg_angles in angles]
                ankles = [leg_angles[2] for leg_angles in angles]
                # row += interleave_with(hips + knees + ankles, 0.0)
                row += interleave_with(list(map(degrees, hips + knees + ankles)), 0.0)

                # Write touch sensors
                row += touches

                writer.writerow(row)


# FLH_1,FLH_2,FRH_1,FRH_2,RLH_1,RLH_2,RRH_1,RRH_2,FLK_1,FLK_2,FRK_1,FRK_2,RLK_1,RLK_2,RRK_1,RRK_2,FLA_1,FLA_2,FRA_1,FRA_2,RLA_1,RLA_2,RRA_1,RRA_2

if __name__ == "__main__":

    kinematics_path = Path("MotionData/")
    animations_path = Path("Animations/")

    animat_config_file = "dog_config.json"
    animat = QuadrupedAnimat(file=animat_config_file)

    with open(animat_config_file, "r") as config_file:
        config_file = json.load(config_file)
        gaits = config_file["gaits"]

    num_gait_cyles = 3
    num_foot_steps = 20

    for gait in gaits:
        print("Running gait:", gait["name"])
        animat.run_gait(
            gait, kinematics_path, animations_path, num_gait_cyles, num_foot_steps
        )
