from math import atan2, cos, degrees, pi, sin, sqrt
from icecream import ic
from typing import Tuple


class Pt:
    def __init__(self, x: float, y: float) -> None:
        """
        Sets the x and y positions based on parameters
        """
        self.x, self.y = x, y

    def __sub__(self, other):
        """
        subtracts another point from current, returns tuple =
        (diff_x, diff_y)
        """
        # print(
        #     f"type self = {type(self)} | {(type(self.x), type(self.y))}, while other is {type(other)} | {(type(other.x), type(other.y))}"
        # )
        # ic(self)
        # ic(other)
        return Pt(self.x - other.x, self.y - other.y)

    def __str__(self) -> str:
        """
        Returns a string representation that lists x and y positions

        Returns:
            str: f"({self.x:.3f}, {self.y:.3f})"
        """
        return f"({self.x:.3f}, {self.y:.3f})"

    def __repr__(self) -> str:
        return self.__str__()

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    def beyond(self, other) -> bool:
        """returns true if other is horizontally past self

        Args:
            other (Pt): another point

        Returns:
            bool: true if self.x > other.x
        """
        return self.x > other.x

    # equiv to "static" in C or Java
    @classmethod
    def angle_between(cls, pt1, pt2) -> float:
        """computes simplified angle between two points

        Args:
            cls (Pt): the class of object; important for classmethod
            pt1 (Pt): first point being compared
            pt2 (Pt): second point being compared

        Returns:
            float: simplified angle representing distance between the two points
        """
        pt1_angle = atan2(pt1.y, pt1.x)
        pt2_angle = atan2(pt2.y, pt2.x)
        return Pt.simplify_angle((pt2_angle - pt1_angle) % (2 * pi))

    @classmethod
    def from_tuple(cls, tuple: Tuple[float, float]):
        """Creates a point object from a tuple

        Args:
            tuple (Tuple[float, float]): [description]

        Returns:
            [type]: [description]
        """
        x, y = tuple
        return Pt(x, y)

    def simplify_angle(angle: float) -> float:
        """
        Simplifies by reducing the size of the angle.

        Args:
            angle (float): initial angle to be simplified

        Returns:
            float: simplified angle
        """
        angle = angle % (2.0 * pi)
        if angle < -pi:
            angle += 2.0 * pi
        elif angle > pi:
            angle -= 2.0 * pi
        return angle
