from __future__ import annotations

import matplotlib.pyplot as plt

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

        self.goal = None
        self.poses = None

        self.states = []

    def add_goal(self, goal: float):
        # TODO
        self.goal = 6
        self.calculate_poses(Gait.WALK)

    def calculate_poses(self, gait: Gait):
        # TODO
        self.poses = [Pt(1, 1), Pt(1, -1)]

    def get_leg_positions(self, start: Pt) -> List[Pt]:

        x, y = (start.x, start.y)
        positions = [Pt(x, y)]

        for seg in self.segs:
            x -= seg.end.x
            y -= seg.end.y
            positions.append(Pt(x, y))

        return positions

    # def move(self, footstate: FootState):
    def move(self) -> int:
        # returns an id if needs to translate

        if self.i >= len(self.poses):
            # restart
            self.i = 0

        goal = self.poses[self.i]
        self.i += 1

        for seg in reversed(self.segs):

            to_tip = self.foot - seg.end
            to_goal = goal - seg.end

            new_angle = Pt.angle_between(to_tip, to_goal)
            seg.set_new_angle(new_angle + seg.angle)

        if goal.y == -1:
            # touched down -> translate!
            return True
        else:
            return False

    @classmethod
    def equal_len_segs(cls: Leg, total_len: float, num_segs: int, hip: Pt) -> Leg:
        """Create a new leg by making equal length segments
        Args:
            cls (Leg): [description]
            total_len (float): [description]
            num_segs (int): [description]
            x_delta (float): [description]
            y_delta (float): [description]
            goal (float, optional): [description]. Defaults to None.
            save_state (bool, optional): [description]. Defaults to False.
        Returns:
            Leg: new leg with equal length segments.
        """

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
