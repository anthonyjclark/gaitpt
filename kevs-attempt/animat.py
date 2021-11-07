from hinged_segment import HingedSegment  # stops the errors, remove later
from point import Pt
from leg import Leg
from gaits import Gait

from typing import List, Tuple, Optional, Union
from __future__ import annotations

from math import atan2, cos, degrees, pi, sin, sqrt


class Animat(object):
    """Represents a 4-legged, double-hipped animal.

    Responsible for:
    - given a goal position, assign sub-goals to each leg for each step necessary
    - cycle through all progress points needed for each leg to reach goal states
    - collect joint angles for each progress point for each leg
    - return joint angles for Coordinator

    """

    def __init__(
        self, initial_x: float, x_delta: float, y_delta: float, ground_y: float
    ):

        super().__init__()

        self.x_delta: float = x_delta
        self.y_delta: float = y_delta
        self.ground_y: float = ground_y  # TODO: delete? we could just assume 0..

        # TODO: make these passable as parameters?
        self.height: float = 3
        self.length: float = 5
        self.num_leg_segs: int = 4
        self.num_legs: int = 4
        self.gait: Gait = None

        # TODO: add support for multiple hips in a list - leading hip will be end of list
        # the base of the animat will be its two hips - ground assumed to be at 0
        self.back_hip: Pt = Pt(initial_x, self.height)  # back legs are at initialx
        self.front_hip: Pt = Pt(initial_x + self.length, self.height)

        self.hips = [self.front_hip, self.back_hip]

        self.legs: List(Leg) = []  # for legs, order is: fl, fr, bl, br

        # adding all legs to animat
        curr_hip: int = 0
        curr_legsonhip: int = 0
        for _ in range(0, self.num_legs):

            new_leg = Leg.equal_len_segs(
                self.height,
                self.num_leg_segs,
                self.hips[curr_hip],
                self.x_delta,
                self.y_delta,
                save_state=True,
            )  # added with no current goal

            self.legs.append(new_leg)

            curr_legsonhip += 1

            if curr_legsonhip > 1:
                curr_hip += 1
                curr_legsonhip = 0

    def assign_gait(self, gait: Gait):
        self.gait = gait

    def update_pos(self):
        self.pos = self.legs[0].get_pos() # TODO: should this be repr by smth else?

    def move(self, goal):

        # strategy: assign sub-goals to each leg, run through those and return the joint positions
