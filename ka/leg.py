from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from unittest.mock import Mock

from typing import List, Tuple, Optional, Union
from icecream import ic

from math import atan2, cos, degrees, pi, sin, sqrt

# from animat import Actors
from hinged_segment import HingedSegment  # stops the errors, remove later
from point import Pt
from gaits import Gait, FootState


class Leg(object):
    def __init__(self, segments: List[HingedSegment], id: int):
        super().__init__()

        self.i = 0
        self.segs = segments
        self.id = id

        self.foot = segments[-1].end
        self.foot_pos = None

        self.goal = None
        self.poses = None

        self.states = []

        self.num_per_step = 10

    def add_goal(self, goal: float):
        # TODO: allow for other gaits
        self.goal = 6
        self.calculate_poses(Gait.WALK.value)

    def calculate_poses(self, gait: List[List[FootState]]):
        # TODO: entirely design

        # v3:
        poses = []
        poses.append(self.foot_pos)

        steps: List[FootState] = gait[
            self.id
        ]  # gets the leg positions for that one leg

        for step in steps:

            # needs to catch errors with multiple steps etc
            if step == FootState.STEP:
                # goes forward, then back
                poses += self.get_step_arc(self.foot_pos, self.num_per_step)

                # should end up back at start

            # elif step == FootState.GROUND:
            #     stay_poses = [self.foot] * self.num_per_step
            #     poses += stay_poses

        self.poses = poses
        print(self.poses)

    def get_leg_positions(self, start: Pt) -> List[Pt]:
        """because a leg only stores relative positions, hip will pass in the actual
        position, and then this function can compute the actual points for the actors to use

        Args:
            start (Pt): location of Hip assigned to this leg

        Returns:
            List[Pt]: actual positions of each leg joint
        """

        x, y = (start.x, start.y)
        positions = [Pt(x, y)]

        for seg in self.segs:
            x += seg.end.x
            y -= seg.end.y
            positions.append(Pt(x, y))

        self.foot_pos = positions[-1]  # updates this every time

        return positions

    def move(self) -> bool:
        """moves leg by implementing reverse kinematics algo used by prof Clark.
        after move, decided whether the animat needs to translate based on whether 
        the leg touched down.

        Returns:
            bool: whether a foot has touched the ground
        """

        if self.i >= len(self.poses):
            # restart
            self.i = 0

        goal = self.poses[self.i]
        self.i += 1

        max_steps = 100

        print(f"current goal for leg {self.id} is {goal}")

        for seg in reversed(self.segs):

            to_tip = self.foot - seg.end
            to_goal = goal - seg.end

            new_angle = Pt.angle_between(to_tip, to_goal)
            seg.set_new_angle(new_angle + seg.angle)

    def get_step_arc(self, start: Pt, num_steps=10) -> List[Pt]:
        # just a triangle for now

        pts: List[Pt] = [start]

        horiz_reach = 1
        vertical_reach = 0.4

        x, y = start.x, start.y
        delta_x = horiz_reach / num_steps
        delta_y = 2 * vertical_reach / num_steps

        for step in range(num_steps):
            x += delta_x
            y += (
                delta_y if step < num_steps // 2 else -delta_y
            )  # up if halfway through, else down
            pts.append(Pt(x=x, y=y))

        # Backward motion path
        for _ in range(int(num_steps * 1.5)):
            x -= delta_x
            pts.append(Pt(x=x, y=y))

        # Path back to initial position
        for step in range(num_steps // 2):
            x += delta_x
            y += delta_y if step < num_steps // 4 else -delta_y
            pts.append(Pt(x=x, y=y))

        return pts

    @classmethod
    def equal_len_segs(cls: Leg, total_len: float, num_segs: int, hip: Pt) -> Leg:
        # TODO: should create legs with equal lenght segments. not currently used

        assert num_segs > 0
        assert total_len > 0.0

        seg_len: float = total_len / num_segs

        segments: List(HingedSegment) = []

        # first segment needs to be connected to hip
        # last segment will be 90 degrees - that's the foot. Others will be at 0
        first_seg = HingedSegment(0.0, seg_len, hip)
        segments.append(first_seg)
        num_segs -= 1

        if num_segs > 1:
            for seg in range(1, num_segs - 1):
                # make a hingedsegment for each of these, all with 0 degree angles (straight down)
                # each will have their parent be the last segment added to our list
                seg = HingedSegment(0.0, seg_len, segments[-1])
                segments.append(seg)

        # now for the last segment - the foot
        last_seg = HingedSegment(90.0, seg_len, segments[-1])
        segments.append(last_seg)

        return Leg(hip, segments)
