from hinged_segment import HingedSegment  # stops the errors, remove later
from point import Pt

from typing import List, Tuple, Optional, Union
from __future__ import annotations

from math import atan2, cos, degrees, pi, sin, sqrt


class Leg(object):
    def __init__(
        self,
        hip: Pt,
        segments: List(HingedSegment),
        x_delta: float,
        y_delta: float,
        goal: float = None,
        save_state: bool = False,
    ) -> None:

        super().__init__()  # inherits from object

        assert len(segments) > 0  # otherwise we don't have a leg!

        self.states = []  # will return these later
        self.hip = hip
        self.x_delta = x_delta
        self.y_delta = y_delta
        self.goal = goal
        self.save_state = save_state
        self.segments = segments

        self.pos = segments[
            -1
        ].get_tip_location()  # position of foot is considered to be position
        self.last_move = self.pos

    def move(self) -> Optional[list]:
        # moves one bit in the process - maybe not a full step, just as far as its deltas let it
        # returns the list of states it took to get here, if it is done

        assert self.goal != None  # otherwise we don't know where to move!

        for seg in reversed(self.segments):
            to_effector = self.effector - seg.loc
            to_goal = self.goal - seg.loc

            new_angle = Pt.angle_between(to_effector, to_goal)
            seg.set_new_angle(new_angle + seg.ang)
            self.effector = self.segments[-1].get_tip_location()

            # Check for termination by comparing new x,y to goal
            reached_goal = True if self.effector == self.goal else False

            # Check if still making progress
            last_x, _ = self.last_move
            curr_x, _ = self.effector
            making_progress = True if curr_x > last_x else False

            # TODO: constraint to specific axis
            # TODO: add joint limits
            # TODO: add direction of effector
            # epsilon = 0.0001
            # trivial_arc_length = 0.00001

            if self.save_state:
                self.add_state()

            if reached_goal or not making_progress:
                # these are the cases when animat needs to assign a new action
                return self.states
            # we don't want to return something every time - just when we're done moving

    def add_state(self):
        """add a state for each segment containing info about their x,y coords and append that sublist
        to the main self.states value
        """
        step_states = []
        for seg in self.segments:
            step_states.append((seg.loc.x, seg.loc.y))
        step_states.append(
            (self.effector.x, self.effector.y)
        )  # effector = tip of last segment in chain
        self.states.append(step_states)
        return step_states

    def get_states(self) -> List[List[Tuple[float, float]]]:
        """self.states contains list of all states we've collected so far. this returns them all

        Returns:
            List[List[Tuple[float, float]]]: all states we've captured so far
        """
        return self.states

    # def plot(self, goal: Optional[Pt] = None) -> None:
    #     """plots a single frame

    #     Args:
    #         goal (Optional[Pt], optional): goal point, optional if we want to plot it. Defaults to None.
    #     """
    #     # All actuated segments
    #     for p1, p2 in zip(self.segments, self.segments[1:]):
    #         plt.plot([p1.loc.x, p2.loc.x], [p1.loc.y, p2.loc.y])

    #     # Last joint to the end
    #     plt.plot(
    #         [self.segments[-1].loc.x, self.effector.x],
    #         [self.segments[-1].loc.y, self.effector.y],
    #     )

    #     # Goal
    #     if goal:
    #         plt.plot(goal.x, goal.y, "o")

    #     plt.axis("equal")

    def set_goal(self, new_goal: float):
        self.goal = new_goal

    def set_deltas(self, new_x: float = None, new_y: float = None):
        if new_x:
            self.x_delta = new_x
        if new_y:
            self.y_delta = new_y

    @classmethod
    def equal_len_segs(
        cls: Leg,
        total_len: float,
        num_segs: int,
        hip: Pt,
        x_delta: float,
        y_delta: float,
        goal: float = None,
        save_state: bool = False,
    ) -> Leg:
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

        return Leg(hip, segments, x_delta, y_delta, goal, save_state)

    def get_pos(self) -> Pt:
        return self.segments[-1].get_tip_location()

    def get_hip(self) -> Pt:
        return self.hip
