#%%
from __future__ import annotations
from dataclasses import dataclass
from itertools import accumulate
from loguru import logger
from math import atan2, cos, inf, pi, sin, sqrt, radians, degrees
from typing import Iterable, List, Tuple
import numpy as np
import csv
import json
from icecream import ic

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numpy.lib.function_base import angle


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


@dataclass(repr=False)  # don't overwrite repr fx
class Pose:
    point: Point
    angle: float

    def __repr__(self):
        return f"(({self.point.x}, {self.point.y}), {self.angle})"


class Animat:
    def __init__(self, file: str = None) -> None:
        """Implementing a four-legged, walking animat.

        Some dimensions and angles inspired by:
            Towards Dynamic Trot Gait Locomotion—Design, Control, and Experiments with
            Cheetah-cub, a Compliant Quadruped Robot
            by Alexander Spröwitz, Alexandre Tuleu, Massimo Vespignani, Mostafa Ajallooeian,
            Emilie Badri, Auke Jan Ijspeert
        """

        if file:
            # create from specifications
            f = open(file)
            f = json.load(f)
            data = f["animat"]

            back_hip = Point(data["hips"][1], data["height"])
            front_hip = Point(data["hips"][0], data["height"])

            self.legs = []

            for leg_dict in data["legs"]:

                angles = []
                for angle in leg_dict["angles"]:
                    # gotta convert all of these to degrees
                    angles.append(deg(angle))

                limits = []
                for limit in leg_dict["limits"]:
                    # each is a 2-len tuple, but in json can only do lists
                    limits.append((deg(limit[0]), deg(limit[1])))

                lengths = leg_dict["lengths"]

                hip = front_hip if leg_dict["hip"] == 0 else back_hip

                self.legs.append(Leg(angles, limits, lengths, hip))

            self.legs[1].raise_hip()
            self.legs[3].raise_hip()

        else:

            back_hip = Point(-0.5, 0)

            # Angle and limits are relative to parent joint (or the world in the case of
            # the firt segment)
            front_leg_angles = [deg(300), deg(-80), deg(80)]
            front_leg_limits = [
                (deg(270), deg(340)),
                (deg(-160), deg(20)),
                (deg(20), deg(160)),
            ]
            front_leg_lengths = [0.26, 0.42, 0.32]
            self.front_left = Leg(
                front_leg_angles, front_leg_limits, front_leg_lengths, hip=Point(0, 0.1)
            )
            self.front_right = Leg(
                front_leg_angles, front_leg_limits, front_leg_lengths
            )

            rear_leg_angles = front_leg_angles
            rear_leg_limits = front_leg_limits
            rear_leg_lengths = [0.41, 0.42, 0.17]

            self.rear_left = Leg(
                rear_leg_angles, rear_leg_limits, rear_leg_lengths, hip=Point(-0.5, 0.1)
            )
            self.rear_right = Leg(
                rear_leg_angles, rear_leg_limits, rear_leg_lengths, hip=back_hip
            )
            self.legs = [
                self.front_left,
                self.front_right,
                self.rear_left,
                self.rear_right,
            ]

    # def _animate(self, positions: list[list[Point]]) -> FuncAnimation:
    def _animate(
        self, frames: List[List[Tuple[List[float], List[float]]]]
    ) -> FuncAnimation:

        fig, ax = plt.subplots()

        lines = [
            ax.plot([], [], marker="o", linewidth=3)[0] for _ in range(len(self.legs))
        ]  # instead of one for each segment, one for each leg
        ax.plot(
            [-1, 1], [frames[0][1][1][-1]] * 2, "k"
        )  # ground - first frame, second leg, rt side of tuple (y), last one (tip)

        def init():
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            return lines

        def update(frame, *fargs) -> Iterable:
            # one frame would have one list of legs, each with an x,y list tuple

            for i, actor in enumerate(lines):
                actor.set_data(
                    frame[i][0], frame[i][1]
                )  # accessing the info in the tuples

            return lines

        animation = FuncAnimation(fig, update, frames=frames, init_func=init)
        return animation

    def walk(self, animate=True) -> list[list[list[Pose]]] | FuncAnimation:
        """Compute joint angles for a walking gait.

        Args:
            animate (bool, optional): return an animation. Defaults to True.

        Returns:
            list[list[Pose]] | FuncAnimation: pose data or animation
        """

        # TODO: vertical reach should go by length of leg
        horiz_reach = 0.6
        vertical_reach = 0.4

        # Initial position given by current leg positions
        positions = [
            [leg.tip_position()] for leg in self.legs
        ]  # should be a list[list[pts]]

        stay_positions = positions  # since they only have a "stay still" goal at first

        # Forward motion path
        num_steps = 16

        xs = [pos[0].x for pos in positions]
        ys = [pos[0].y for pos in positions]

        # prev version
        # delta_x = horiz_reach / num_steps
        delta_y = 2 * vertical_reach / num_steps

        # first do reach
        horiz_reaches = [leg.max_reach for leg in self.legs]
        x_delts = [
            (reach / num_steps) * 0.6 for reach in horiz_reaches
        ]  # full for sprint,

        assert len(xs) == len(ys), "Different x and y lengths for legs"

        for step in range(num_steps):

            for i in range(
                len(self.legs)
            ):  # assuming same lengths, we can do it all here
                # xs[i] += delta_x
                xs[i] += x_delts[i]
                ys[i] += delta_y if step < num_steps // 2 else -delta_y
                positions[i].append(Point(x=xs[i], y=ys[i]))

        # Backward motion path
        for _ in range(int(num_steps * 1.5)):
            for i in range(len(self.legs)):
                # xs[i] -= delta_x
                xs[i] -= x_delts[i]
                positions[i].append(Point(x=xs[i], y=ys[i]))

        # Path back to initial position
        for step in range(num_steps // 2):

            for i in range(len(self.legs)):
                # xs[i] += delta_x
                xs[i] += x_delts[i]
                ys[i] += delta_y if step < num_steps // 4 else -delta_y
                positions[i].append(Point(x=xs[i], y=ys[i]))

        # walk order is: fr, br, fl, bl
        # fl_len = len(positions[0])
        # fr_len = len(positions[1])
        # bl_len = len(positions[2])
        # br_len = len(positions[3])

        # # pad each leg's positions out so that they 'wait' for their turn. imagine 4 chunks, each either step or wait
        # positions[0] = (
        #     stay_positions[0] * fl_len * 2 + positions[0] + stay_positions[0] * fl_len
        # )  # fl
        # positions[1] = positions[1] + stay_positions[1] * fr_len * 3  # fr
        # positions[2] = stay_positions[2] * bl_len * 3 + positions[2]  # bl
        # positions[3] = (
        #     stay_positions[3] * br_len + positions[3] + stay_positions[3] * br_len * 2
        # )  # br

        initial_pts = self.get_pts_from_gjp()

        # for each leg, we're going to run gjp on it, strip only the points out, and then separate the points into tuples of x,y
        frames = [
            [
                self.split_pts(initial_pts[0]),
                self.split_pts(initial_pts[1]),
                self.split_pts(initial_pts[2]),
                self.split_pts(initial_pts[3]),
            ]
        ]  # each frame contains info for one step of all 4 legs

        # Compute joint angles for each point along the path
        # weird structure bc we want them separated by frames and not by leg
        for goal_idx in range(len(positions[0])):  # again, assuming all lens same

            for leg_idx, leg in enumerate(self.legs):
                leg.move_tip(positions[leg_idx][goal_idx])

            if animate:
                all_pts = self.get_pts_from_gjp()

                frames.append(
                    [
                        self.split_pts(all_pts[0]),
                        self.split_pts(all_pts[1]),
                        self.split_pts(all_pts[2]),
                        self.split_pts(all_pts[3]),
                    ]
                )
            else:
                # if we're not animating, we need the full poses
                frame = []
                for leg in self.legs:
                    frame.append(leg.global_joint_poses())
                frames.append(frame)

        return self._animate(frames) if animate else frames

    def do_job(self, job_dict: dict):
        # perform job according instruction dict
        vertical_reach = 0.4
        hor_reach = job_dict["reach multiplier"]
        foot_order = job_dict["foot order"]

        # Initial position given by current leg positions
        positions = [
            [leg.tip_position()] for leg in self.legs
        ]  # should be a list[list[pts]]

        stay_positions = positions  # since they only have a "stay still" goal at first

        # Forward motion path
        num_steps = 16

        xs = [pos[0].x for pos in positions]
        ys = [pos[0].y for pos in positions]

        delta_y = 2 * vertical_reach / num_steps

        # first do reach
        horiz_reaches = [leg.max_reach for leg in self.legs]
        x_delts = [
            (reach / num_steps) * job_dict["reach multiplier"]
            for reach in horiz_reaches
        ]  # full for sprint,

        for curr_ft in foot_order:
            # only some legs are moving at this time

            for step in range(num_steps):

                for i in range(
                    len(self.legs)
                ):  # assuming same lengths, we can do it all here
                    # xs[i] += delta_x
                    if i in curr_ft:
                        xs[i] += x_delts[i]
                        ys[i] += delta_y if step < num_steps // 2 else -delta_y
                        positions[i].append(Point(x=xs[i], y=ys[i]))
                    else:
                        # not moving, just use the last position
                        positions[i].append(positions[i][-1])

            # Backward motion path
            for _ in range(int(num_steps * 1.5)):
                for i in range(len(self.legs)):
                    if i in curr_ft:
                        xs[i] -= x_delts[i]
                        positions[i].append(Point(x=xs[i], y=ys[i]))
                    else:
                        positions[i].append(positions[i][-1])
            # Path back to initial position
            for step in range(num_steps // 2):

                for i in range(len(self.legs)):
                    if i in curr_ft:
                        xs[i] += x_delts[i]
                        ys[i] += delta_y if step < num_steps // 4 else -delta_y
                        positions[i].append(Point(x=xs[i], y=ys[i]))
                    else:
                        positions[i].append(positions[i][-1])

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

        save_frames = [
            [
                self.legs[0].global_joint_poses(),
                self.legs[1].global_joint_poses(),
                self.legs[2].global_joint_poses(),
                self.legs[3].global_joint_poses(),
            ]
        ]

        # Compute joint angles for each point along the path
        # weird structure bc we want them separated by frames and not by leg
        for goal_idx in range(len(positions[0])):  # again, assuming all lens same

            for leg_idx, leg in enumerate(self.legs):
                leg.move_tip(positions[leg_idx][goal_idx])

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
            frame = []
            for leg in self.legs:
                frame.append(leg.global_joint_poses())
            save_frames.append(frame)

        animation = self._animate(anim_frames)
        HTML(animation.to_jshtml())
        animation.save(f'{job_dict["name"]}.gif')

        save_data(save_frames, f'{job_dict["name"]}.csv')

    def canter_job(self, job_dict: dict):
        # specifically for trot => max two legs on the ground, no suspend state
        # when a leg is moving backwards, we'll have the other legs either start their step or just lift a little
        vertical_reach = 0.4
        hor_reach = job_dict["reach multiplier"]
        foot_order = job_dict["foot order"]

        # Initial position given by current leg positions
        positions = [
            [leg.tip_position()] for leg in self.legs
        ]  # should be a list[list[pts]]

        stay_positions = positions  # since they only have a "stay still" goal at first

        # Forward motion path
        num_steps = 16

        xs = [pos[0].x for pos in positions]
        ys = [pos[0].y for pos in positions]

        delta_y = 2 * vertical_reach / num_steps

        # first do reach
        horiz_reaches = [leg.max_reach for leg in self.legs]
        x_delts = [
            (reach / num_steps) * job_dict["reach multiplier"]
            for reach in horiz_reaches
        ]

        # 0 = staging, 1 = forward, 2 = back, 3 = reposition, 4 = done
        stages = [0] * len(self.legs)
        ic(stages)
        fo_idx = 0  # to cycle through the feet that need to move

        while np.min(stages) <= 3:

            # load up the next foot order, if there is one
            ic(stages)
            # ic(foot_order[fo_idx])

            if fo_idx < len(foot_order):

                for leg_idx in foot_order[fo_idx]:
                    # these need to start stepping forward
                    stages[leg_idx] = 1

                fo_idx += 1
            ic(stages)

            for i, leg in enumerate(self.legs):
                # for each leg, check what stage they're in and add the appropriate positions
                if stages[i] == 0 or stages[i] == 4:
                    # staging, don't move forward automatically
                    for step in range(num_steps):
                        positions[i].append(positions[i][-1])  # just stay still
                elif stages[i] == 1:
                    # needs to step forward all the way
                    for step in range(num_steps):
                        xs[i] += x_delts[i]
                        ys[i] += delta_y if step < num_steps // 2 else -delta_y
                        positions[i].append(Point(x=xs[i], y=ys[i]))
                    stages[i] += 1
                elif stages[i] == 2:
                    # move backward
                    for step in range(int(num_steps * 1.5)):
                        xs[i] -= x_delts[i]
                        positions[i].append(Point(x=xs[i], y=ys[i]))
                    stages[i] += 1
                elif stages[i] == 3:
                    # reposition
                    for step in range(num_steps // 2):
                        xs[i] += x_delts[i]
                        ys[i] += delta_y if step < num_steps // 4 else -delta_y
                        positions[i].append(Point(x=xs[i], y=ys[i]))
                    stages[i] += 1

            # if already moving, go to next stage. if waiting, keep waiting until foot order says otherwise
            # stages = list(map(lambda x: x + 1 if x > 0 and x < 4 else x, stages))

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

        save_frames = [
            [
                self.legs[0].global_joint_poses(),
                self.legs[1].global_joint_poses(),
                self.legs[2].global_joint_poses(),
                self.legs[3].global_joint_poses(),
            ]
        ]

        # Compute joint angles for each point along the path
        # weird structure bc we want them separated by frames and not by leg
        for goal_idx in range(len(positions[0])):  # again, assuming all lens same

            for leg_idx, leg in enumerate(self.legs):
                leg.move_tip(positions[leg_idx][goal_idx])

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
            frame = []
            for leg in self.legs:
                frame.append(leg.global_joint_poses())
            save_frames.append(frame)

        animation = self._animate(anim_frames)
        HTML(animation.to_jshtml())
        animation.save(f'{job_dict["name"]}.gif')

        save_data(save_frames, f'{job_dict["name"]}.csv')

    def split_pts(self, pts: List[Point]) -> Tuple[List[float], List[float]]:
        # helper function, since we can update an actor with all x and y coordinates in this format
        xs = []
        ys = []
        for pt in pts:
            xs.append(pt.x)
            ys.append(pt.y)

        return (xs, ys)

    def get_pts_from_gjp(self) -> List[List[Point]]:
        # runs global_joint_poses on each leg in self.legs and returns a list of points for each leg
        all_pts = []

        for leg in self.legs:

            leg_pts = [pose.point for pose in leg.global_joint_poses()]
            all_pts.append(leg_pts)

        return all_pts


class Leg:
    def __init__(
        self,
        angles: list[float],
        limits: list[tuple[float, float]],
        lengths: list[float],
        hip: Point = Point(),  # 0,0 by default
    ) -> None:
        """Create a single leg.

        Args:
            angles (list[float]): list of angles relative to their preceding angle
            limits (list[tuple[float, float]]): list of angle limits relative to parent joint
            lengths (list[float]): list of leg lengths
        """

        self.num_segments = 3
        self.hip = hip

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
        poses = [Pose(self.hip, self.angles[0])]

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

            # Check if close enough to goal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               l
            dist = (goal - self.tip_position()).norm
            if abs(dist) < tolerance:
                break

            # Check if still making progress (goal might be out of reach)
            if abs(dist - prev_dist) < tolerance:
                break

            prev_dist = dist

    def raise_hip(self):
        # just raises the hip a tiny bit so that we can see all 4 legs in animation
        self.hip = Point(self.hip.x, self.hip.y + 0.1)


def save_data(data: List[List[List[Pose]]], filename: str):
    # writes data to file
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        for frame in data:
            # each frame is a row
            arr = np.array(frame)
            writer.writerow(arr)


# animat = Animat()
# animation = animat.walk()
# training_data = animat.walk(animate=False)
# save_data(training_data, "walk_example.csv")
# HTML(animation.to_jshtml())
# animation.save("example.gif")

# test creation from json file
animat2 = Animat(file="sample_json.json")

with open("sample_json.json", "r") as f:
    f = json.load(f)
    jobs = f["jobs"]

    for job in jobs:
        if job["name"] == "canter":
            animat2.canter_job(job)
        else:
            # animat2.do_job(job)
            pass

# animation2 = animat2.walk()
# training_data = animat2.walk(animate=False)
# save_data(training_data, "walk_example.csv")
# HTML(animation2.to_jshtml())
# animation2.save("example2.gif")

# %%
