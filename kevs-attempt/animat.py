from __future__ import annotations
from hinged_segment import HingedSegment  # stops the errors, remove later
from point import Pt
from leg import Leg
from gaits import Gait, FootState

from typing import List, Tuple, Optional, Union


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

        self.body = self.midpoint(self.front_hip, self.back_hip)

        self.hips = [self.front_hip, self.back_hip]

        self.legs: List(Leg) = []  # for legs, order is: fl, fr, bl, br
        self.leg_angles = []  # same length and order as self.legs

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
        # midpoint between the hips
        self.front_hip = self.legs[0].get_hip()
        self.back_hip = self.legs[2].get_hip()
        self.pos = self.midpoint(self.front_hip, self.back_hip)
        return self.pos

    def get_pos(self):
        return self.pos

    def move(self, goal):

        assert self.gait != None

        print(f"legs = {self.legs}, gait = {self.gait}")
        # strategy: assign sub-goals to each leg, run through those and return the joint positions
        assert len(self.legs) == len(self.gait)  # TODO: allow for repeating values

        curr_gait = 0
        legs_done = [0] * len(self.legs)

        # one while loop until you get to the goal. one loop for cycling through the legs and applying their gait
        while self.update_pos()[0] < goal.x:

            if min(legs_done) > 0:
                # no 0's therefore all legs done. next gait is up. we need to circle around sometimes
                curr_gait = curr_gait + 1 if curr_gait < len(self.legs) - 1 else 0
                legs_done = [0] * len(self.legs)  # reset!

            for idx, leg in enumerate(self.legs):
                # we've already verified that self.legs and self.gait have same length, so apply 1:1

                # if they already have a goal assigned, keep moving and wait for it to return a set of positions
                if leg.has_goal() and not legs_done[idx] > 0:
                    # move both performs the move and optionally returns a list if it is done. we check if the list
                    # was returned so that we know when to assign it a new goal.
                    leg_angles = leg.move()
                    if leg_angles != None:
                        # it finished! we need to provide it a new goal for next time around and mark that it's done
                        legs_done[idx] = 1

                        leg.set_goal(None)
                        # leg.set_goal(
                        #     self.new_leg_goal(
                        #         goal,
                        #         self.gait[curr_gait][idx],
                        #         (leg.x_delta, leg.y_delta),
                        #     )
                        # )

                        self.leg_angles[idx].append(
                            leg_angles
                        )  # appends to the list for that particular leg

                    # if no list returned, we're not done moving this leg. keep going on the others, we'll come back around
                elif legs_done[idx] > 0:
                    # this leg has finished its movement.
                    continue
                else:
                    # no goal, so let's calculate what the goals would be for each.
                    leg.set_goal(
                        self.new_leg_goal(
                            goal, self.gait[curr_gait][idx], (leg.x_delta, leg.y_delta)
                        )
                    )

    def new_leg_goal(self, overall_goal, gait, leg_deltas) -> Pt:

        # TODO: set parameter for ground? for now assume y=0 is ground
        ground_y = 0

        new_goal: Pt = (0, 0)

        y_delt, x_delt = leg_deltas
        curr_x, curr_y = overall_goal.x, overall_goal.y

        if gait == FootState.GROUND:
            # if ground, same as start, except to move down
            if curr_y > ground_y:
                # probably in the middle of a step. continue the step
                new_x = curr_x + x_delt / 2

            else:
                # just holding steady
                new_x = curr_x

            new_y = ground_y
            new_goal = (curr_x, new_y)
        elif gait == FootState.SUSPEND:
            # if suspend, move forward one half delta, not vertically at all (unless to move slightly up)
            new_y = (
                curr_y if curr_y > ground_y else y_delt / 2
            )  # just a tiny shift upwards if touching ground
            new_x = curr_x + x_delt
            new_goal = (new_x, new_y)
        elif gait == FootState.STEP:
            # takes a step upwards, initiating the swing
            new_x = curr_x + x_delt / 2
            new_y = ground_y
            new_goal = (new_x, new_y)

        return new_goal

    def get_angles(self):
        return self.leg_angles

    def midpoint(self, pt1, pt2):
        return ((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2)

    def get_length(self):
        return self.length
