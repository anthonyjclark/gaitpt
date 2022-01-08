from __future__ import annotations
import matplotlib.pyplot as plt

from unittest.mock import Mock

from typing import List, Tuple, Optional, Union
from icecream import ic

from math import atan2, cos, degrees, pi, sin, sqrt

from point import Pt
from gaits import Gait, FootState


def clip(val: float, lo: float, hi: float) -> float:
    return max(min(val, hi), lo)


class HingedSegment(object):
    def __init__(
        self, start: Pt, start_angle: float, max_angles: Tuple[float, float], len: float
    ):
        """creates a HingedSegment, which represents relative positions of the leg. no parent; we instead have
        each segment assume it starts at 0,0 and allow the Hip to interpret their positions

        Args:
            start (Pt): #TODO: remove
            angle (float): angle relative to the ground, with straight down being 0 degrees
            len (float): segment length
        """
        self.start = start  # hardcoding this one. probably want to keep this behavior
        self.angle = start_angle
        self.len = len

        # to the right are negative values, to the left are positive
        self.max_angles = max_angles

        self.end = self.calculate_end()

        self.chi = None

    def set_max_angles(self, low: float, high: float):
        self.max_angles[1] = low
        self.max_angles[0] = high

    def calculate_end(self) -> Pt:
        """calculates position of the tip of this segment

        Returns:
            Pt: the tip
        """
        x = self.start.x + (self.len * sin(self.angle))
        y = self.start.y + (self.len * cos(self.angle))
        return Pt(x, y)

    def add_child(self, chi: HingedSegment):
        """adds a segment in as a child, linkedlist style

        Args:
            chi (HingedSegment): ONLY hingedsegments can be connected
        """
        self.chi = chi

    def update_from_parent(self, delta_angle: float) -> None:
        """propagates changes in angle from parent HS to child HS objects

        Args:
            delta_angle (float): amount by which angle will change
        """
        self.angle += delta_angle
        self.end = self.calculate_end()
        if self.chi:
            # update child if present
            self.chi.update_from_parent(delta_angle)

    def set_new_angle(self, new_angle: float) -> None:
        """change self angle to parameter and propagate that change through children

        Args:
            new_angle (float): new angle for this segment
        """

        new_angle = clip(new_angle, self.max_angles[1], self.max_angles[0])

        delta_angle = (
            new_angle - self.angle
        )  # the difference between new and old angles
        self.angle = new_angle
        self.end = self.calculate_end()
        if self.chi:
            self.chi.update_from_parent(delta_angle)

    def get_tip(self) -> Pt:
        """wrapper function; calls calculate_end and returns the result, also updating self.end

        Returns:
            Pt: coordinates of tip
        """
        self.end = self.calculate_end()
        return self.end
