# from __future__ import annotation
import matplotlib.pyplot as plt
from os import chdir
from unittest.mock import Mock

from typing import List, Tuple, Optional, Union
from icecream import ic

from math import atan2, cos, degrees, pi, sin, sqrt

from hinged_segment import HingedSegment
from hip import Hip  # stops the errors, remove later
from point import Pt
from leg import Leg
from gaits import Gait, FootState

Actors = List[plt.Line2D]


def check_called(func):
    return Mock(side_effect=func)


class Animat(object):
    def __init__(self, num_legs: int, num_segs: int, height: float, length: float):
        self.num_legs = num_legs
        self.num_segs = num_segs
        self.ht = height
        self.length = length

        self.goal = None

        # TODO: hip id programmatically
        # create both hips - front to back
        back_hip_pos = Pt(0, self.ht)
        self.back_hip = Hip(back_hip_pos, 1)
        front_hip_pos = Pt(self.length, self.ht)
        self.front_hip = Hip(front_hip_pos, 0)
        self.hips = [self.front_hip, self.back_hip]

        # TODO: fix leg creation methods
        leg_specs = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        # create all the legs - front to back
        self.legs: List[Leg] = self.create_legs(hips=self.hips, specs=leg_specs)

        # one row for each leg, with a list containing all the steps that
        # leg has taken, with each step being a list of positions for each joint
        self.leg_data = [[]] * self.num_legs

    def add_goal(self, goal: float):

        self.goal = goal

        for leg in self.legs:
            leg.add_goal(goal)

    def move(self):

        assert self.goal

        for i, hip in enumerate(self.hips):
            if hip.move():
                # if needs to translate
                self.translate(i)

        # TODO: we want to concatenate these lists properly. each should have two lists
        states = []
        for hip in self.hips:
            states += hip.get_leg_positions()

        self.leg_data.append(states)

    def get_last_step_data(self) -> List[List[Pt]]:
        return self.leg_data[-1]

    def translate(self, idx: int):
        for i in range(idx, len(self.hips)):
            self.hips[i].translate()

    def create_legs(
        self,
        hips: List[Pt],
        specs: List[List[float]] = None,
        ht: float = None,
        num_legs: int = None,
        num_segs: int = None,
    ):
        # hips always needs to be passed in
        assert len(hips) > 0

        if hips and specs:
            return self.create_legs_from_spec(hips, specs)
        elif hips and ht and num_legs and num_segs:
            return self.create_equal_legs(hips, ht, num_legs, num_segs)
        else:
            raise ValueError(
                "Insufficient arguments. Use either hips and specs or hips, height, num legs, and num segs"
            )

    # @check_called
    def create_legs_from_spec(
        self, hips: List[Hip], seg_specs: List[List[float]]
    ) -> List[Leg]:
        """creates legs for animat from specs

        Args:
            actors (Actors): [description]
            ht (float): [description]
            seg_specs (List[List[float]]): one column for each leg. each has list of values for all segments inside
            hips (List[Pt]): [description]

        Returns:
            List[Leg]: [description]
        """

        legs = []
        default_globalangle = 0.0  # maybe needs to change for some animals, but not rn

        # TODO: not hardcoded
        segs1 = [
            HingedSegment(Pt(0, 0), default_globalangle, 1.0),
            HingedSegment(Pt(0, 0), default_globalangle, 1.0),
            HingedSegment(Pt(0, 0), default_globalangle, 1.0),
            HingedSegment(Pt(0, 0), default_globalangle, 1.0),
        ]
        segs2 = [
            HingedSegment(Pt(0, 0), default_globalangle, 1.1),
            HingedSegment(Pt(0, 0), default_globalangle, 1.1),
            HingedSegment(Pt(0, 0), default_globalangle, 1.1),
            HingedSegment(Pt(0, 0), default_globalangle, 1.1),
        ]
        segs3 = [
            HingedSegment(Pt(0, 0), default_globalangle, 1.0),
            HingedSegment(Pt(0, 0), default_globalangle, 1.0),
            HingedSegment(Pt(0, 0), default_globalangle, 1.0),
            HingedSegment(Pt(0, 0), default_globalangle, 1.0),
        ]
        segs4 = [
            HingedSegment(Pt(0, 0), default_globalangle, 1.1),
            HingedSegment(Pt(0, 0), default_globalangle, 1.1),
            HingedSegment(Pt(0, 0), default_globalangle, 1.1),
            HingedSegment(Pt(0, 0), default_globalangle, 1.1),
        ]

        l0 = Leg(segs1, 0)
        l1 = Leg(segs2, 1)
        l2 = Leg(segs3, 2)
        l3 = Leg(segs4, 3)

        legs = [l0, l1, l2, l3]
        hips[0].set_legs([l0, l1])
        hips[1].set_legs([l2, l3])

        # # TODO: let hips decide how many it can hold
        # legs_on_hips = [0 * len(hips)]  # repr # of legs on each hip
        # curr_hip = 0
        # max_legs_per_hip = len(seg_specs) / len(hips)

        # for i, _ in enumerate(seg_specs):
        #     # these are all legs
        #     leg_segs = []
        #     prev_seg = None

        #     hip = hips[curr_hip]
        #     legs_on_hips[curr_hip] += 1
        #     if legs_on_hips[curr_hip] >= max_legs_per_hip:
        #         curr_hip += 1

        #     for j, length in enumerate(seg_specs[i]):
        #         # these are segments of a leg
        #         hs = HingedSegment(Pt(0, 0), default_globalangle, length)  # TODO
        #         leg_segs.append(hs)

        #         if prev_seg:
        #             prev_seg.add_child(hs)

        #         prev_seg = hs

        #     leg = Leg(leg_segs, i)
        #     hips[curr_hip].add_leg(leg)
        #     legs.append(leg)

        return legs

    @check_called
    def create_equal_legs(hips, ht, num_legs, num_segs):

        legs = []

        legs_on_hips = [0 * len(hips)]  # repr # of legs on each hip
        curr_hip = 0
        max_legs_per_hip = num_legs / len(hips)

        for i in range(num_legs):
            hip = hips[curr_hip]

            legs.append(Leg.equal_len_segs(ht, num_segs, hip))

            legs_on_hips[curr_hip] += 1
            if legs_on_hips[curr_hip] >= max_legs_per_hip:
                curr_hip += 1

        return legs

