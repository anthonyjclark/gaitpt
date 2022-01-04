from __future__ import annotations
import matplotlib.pyplot as plt

from unittest.mock import Mock

from typing import List, Tuple, Optional, Union
from icecream import ic

from math import atan2, cos, degrees, pi, sin, sqrt

from point import Pt
from gaits import Gait, FootState


class HingedSegment(object):
    def __init__(self, start: Pt, angle: float, len: float):
        # self.start = start
        self.start = Pt(0, 0)
        self.angle = angle
        self.len = len

        self.end = self.calculate_end()

        self.chi = None

    def calculate_end(self) -> Pt:
        x = self.start.x + self.len * cos(self.angle)
        y = self.start.y + self.len * sin(self.angle)
        return Pt(x, y)

    def add_child(self, chi: HingedSegment):
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
        delta_angle = (
            new_angle - self.angle
        )  # the difference between new and old angles
        self.angle = new_angle
        self.end = self.calculate_end()
        if self.chi:
            self.chi.update_from_parent(delta_angle)

    def get_tip(self):
        return self.end
