from __future__ import annotations
import matplotlib.pyplot as plt

from unittest.mock import Mock

from typing import List, Tuple, Optional, Union
from icecream import ic

from math import atan2, cos, degrees, pi, sin, sqrt

from point import Pt


class Hip(object):
    def __init__(self, pos: Pt, id: int):
        self.id = id
        self.pos = pos
        self.legs = []
        self.states = []
        self.trans = False

    def set_legs(self, legs):
        """adds input legs to internal legs tracker

        Args:
            legs (List[Leg]): [description]
        """
        self.legs = legs
        self.states = []

    def add_leg(self, leg):
        self.legs.append(leg)

    def get_leg_positions(self) -> List[List[Pt]]:
        """Hip stores an absolute position for itself. Legs only store a relative position
        to their respective hip. This function combines that information to get actual
        coordinates for actors to use

        Returns:
            List[List[Pt]]: for each leg, a list of the positions from component segments
        """
        # legs have delta from hip

        actors = []
        print(f" I have {len(self.legs)} legs and am index {self.id}")

        for leg in self.legs:

            actors.append(leg.get_leg_positions(self.pos))

        return actors

    def add_state(self):
        """adds the last set of positions to its internal states
        """
        self.states.append(self.get_leg_positions())

    def get_last_state(self):
        """gets last set of position for use in animation
        """
        return self.states[-1]

    def translate(self):
        # TODO: allow other inputs for xdelta
        self.pos = Pt(self.pos.x + 1, self.pos.y)

    def move(self):

        for leg in self.legs:
            translate = leg.move()
            if translate:
                self.trans = True

        if self.trans:
            return True
        else:
            return False

