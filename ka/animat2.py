from __future__ import annotations
from ka.hinged_segment import HingedSegment  # stops the errors, remove later
from ka.point import Pt
from ka.leg import Leg
from ka.gaits import Gait, FootState
import matplotlib.pyplot as plt

from unittest.mock import Mock

from typing import List, Tuple, Optional, Union
from icecream import ic

from math import atan2, cos, degrees, pi, sin, sqrt

Actors = List[plt.Line2D]


def check_called(func):
    return Mock(side_effect=func)


class Animat(object):
    def __init__(self, actors, num_segs, height, length):
        self.actors = actors
        self.num_legs = len(actors)
        self.num_segs = num_segs
        self.ht = height

        # create all the legs - front to back
        self.legs: List[Leg] = self.create_legs(self.actors, self.ht, self.num_segs)

        # create both hips - front to back
        self.back_hip = Pt(0, self.ht)
        self.front_hip = Pt(self.length, self.ht)
        self.hips = [self.front_hip, self.back_hip]

        # one row for each leg, with a list containing all the steps that
        # leg has taken, with each step being a list of positions for each joint
        self.leg_data = [[]] * self.num_legs

    @check_called
    def create_legs_from_spec(
        seg_specs: List[List[float]], hips: List[Pt]
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

        legs_on_hips = [0 * len(hips)]  # repr # of legs on each hip
        curr_hip = 0
        max_legs_per_hip = len(seg_specs) / len(hips)

        for i, _ in enumerate(seg_specs):
            # these are all legs
            leg_segs = []
            prev_seg = None

            hip = hip[curr_hip]
            legs_on_hips[curr_hip] += 1
            if legs_on_hips[curr_hip] >= max_legs_per_hip:
                curr_hip += 1

            for j, length in enumerate(seg_specs[i]):
                # these are segments of a leg
                if prev_seg:
                    # there exists a previous segment, so it's not the first
                    leg_segs.append(
                        HingedSegment(default_globalangle, length, prev_seg)
                    )
                else:
                    # first segment, have to assign to hip
                    leg_segs.append(HingedSegment(default_globalangle, length, hip))

            leg = Leg(hip, leg_segs)
            legs.append(leg)

        return legs

    @check_called
    def create_equal_legs(ht, num_legs, num_segs, hips):

        legs = []

        legs_on_hips = [0 * len(hips)]  # repr # of legs on each hip
        curr_hip = 0
        max_legs_per_hip = num_legs / len(hips)

        for i in range(num_legs):
            hip = hip[curr_hip]

            legs.append(Leg.equal_len_segs(ht, num_segs, hip))

            legs_on_hips[curr_hip] += 1
            if legs_on_hips[curr_hip] >= max_legs_per_hip:
                curr_hip += 1

        return legs

