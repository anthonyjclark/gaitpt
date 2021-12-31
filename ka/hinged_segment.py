from __future__ import annotations

# from hinged_segment import HingedSegment  # stops the errors, remove later
from ka.point import Pt

from typing import List, Tuple, Optional, Union


from math import atan2, cos, degrees, pi, sin, sqrt


class HingedSegment:
    def __init__(
        self,
        global_angle: float,
        length: float,
        parent_or_location: Union[HingedSegment, Pt],
        max_angle: float = 90.0,
    ) -> None:
        """Takes a global angle, decides if it has a parent, then creates a HS object.

        Args:
            global_angle (float): angle in relation to y axis and ground
            length (float): length of segment
            parent_or_location (Union[HingedSegment, Pt]): either contains a parent HS object, or a starting
                location if no parent is given
        """
        # Angle is always with respect to the global x-y coordinate system
        self.ang = global_angle
        self.len = length

        # For type checking
        self.loc = Pt(0, 0)
        self.par: Optional[HingedSegment]
        self.chi: Optional[HingedSegment] = None
        self.hip = None

        if isinstance(parent_or_location, HingedSegment):
            # if what was passed in is a HS, it is a parent
            self.par = parent_or_location
            self.update_from_parent(0)
        else:
            # here, the parent_or_location var doesn't have type HS, so it can't be a parent.
            #  it contains the location of a hip
            self.par = None
            self.hip = parent_or_location
            # self.loc = parent_or_location

    def __str__(self) -> str:
        """Represents object as string containing info on location and angle

        Returns:
            str: str(self.loc) + f" @ {degrees(self.ang): .3f}°"
        """
        return str(self.loc) + f" @ {degrees(self.ang): .3f}°"

    def __repr__(self) -> str:
        """returns same thing as __str__, for times when str() is not automatically called

        Returns:
            str: [description]
        """
        return self.__str__()

    def update_from_parent(self, delta_angle: float) -> None:
        """propagates changes in angle from parent HS to child HS objects

        Args:
            delta_angle (float): amount by which angle will change
        """
        self.ang += delta_angle
        self.loc.x = self.par.loc.x + self.par.len * cos(self.par.ang)
        self.loc.y = self.par.loc.y + self.par.len * sin(self.par.ang)
        if self.chi:
            # update child if present
            self.chi.update_from_parent(delta_angle)

    def set_new_angle(self, new_angle: float) -> None:
        """change self angle to parameter and propagate that change through children

        Args:
            new_angle (float): new angle for this segment
        """
        delta_angle = new_angle - self.ang  # the difference between new and old angles
        self.ang = new_angle
        if self.chi:
            self.chi.update_from_parent(delta_angle)

    def get_tip_location(self) -> Pt:
        """Gets the x and y location of the tip of each segment

        Returns:
            Pt: position of tip
        """
        tip = Pt(0, 0)
        tip.x = self.loc.x + self.len * cos(self.ang)
        tip.y = self.loc.y + self.len * sin(self.ang)
        return tip

    def get_hip(self):
        if self.hip == None:
            return self.par.get_hip()
        else:
            return self.hip

    def add_child(self, chi: HingedSegment):
        self.chi = chi
