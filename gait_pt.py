#!/usr/bin/env python

# TODO:
# - types
# - update types to Python 3.10 List -> list

#%%
from enum import IntEnum
from typing import List, Tuple, Optional, Union
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from icecream import ic

from IPython.display import HTML

Actors = List[plt.Line2D]
Poses = List[Tuple[float, List[Tuple[float, float]]]]

#%%


class FootState(IntEnum):
    SWING = 1
    LIFT = 2
    THRUST = 3
    SUPPORT = 4

    # TODO: remove this one
    GROUND = 5


# Start with just one leg
def walk_footfalls():
    ...


# %%


class Leg(object):
    def __init__(
        self, stride: float, state: FootState, foot_x: float, swing_y: float
    ) -> None:
        super().__init__()
        self.state = state
        self.stride = stride
        self.half_stride = stride / 2
        self.foot_x = foot_x
        self.swing_y = swing_y

    def is_swinging(self) -> bool:
        return self.state == FootState.SWING

    def not_swinging(self) -> bool:
        return not self.is_swinging()

    def motion_step(self, hip_x: float) -> Tuple[float, float]:

        if self.not_swinging() and self.foot_x < hip_x - self.half_stride:
            self.state = FootState.SWING
        elif self.not_swinging():
            pass
        elif self.is_swinging() and self.foot_x > hip_x + self.half_stride:
            self.state = FootState.GROUND
        else:
            # TODO: make foot delta a parameter
            self.foot_x += 8 / (32 / 3)

        return self.foot_x, self.swing_y if self.is_swinging() else 0

    def create_actors(self, ax: plt.Axes) -> Actors:
        # TODO: linewidth as parameter
        (leg_ln,) = ax.plot([], [], marker="o", linewidth=5)
        return [leg_ln]


class Animat(object):
    # TODO: four legs, give center of body instead of hip
    # - hip height
    def __init__(self, stride: float, initial_x: float, foot_lift: float) -> None:
        super().__init__()

        self.hip_x = initial_x

        x = initial_x  # + stride / 2
        self.legs = [
            Leg(stride, FootState.GROUND, x, foot_lift),  # Back left
            Leg(stride, FootState.SWING, x, foot_lift),  # Back right
        ]

    def motion_step(self, new_x: float) -> Tuple[float, List[Tuple[float, float]]]:

        # TODO: change to body
        self.hip_x = new_x

        # TODO: how to collect pose info for multiple legs and multiple joints? (dict?)
        leg_poses = [leg.motion_step(self.hip_x) for leg in self.legs]

        return self.hip_x, leg_poses

    def generate_poses(self, num_steps: int, final_x: float) -> Poses:
        xs = np.linspace(self.hip_x, final_x, num_steps)
        poses = [self.motion_step(x) for x in xs]
        return poses

    def create_actors(self, ax: plt.Axes) -> Actors:
        # TODO: create body actors

        actors = []
        for leg in self.legs:
            actors.extend(leg.create_actors(ax))

        return actors

    def update_actors(self, actors: Actors, poses: Poses, frame_index: int) -> None:

        hip_x, foot_poses = poses[frame_index]
        for leg_ln, (foot_x, foot_y) in zip(actors, foot_poses):
            leg_ln.set_data([hip_x, foot_x], [1, foot_y])

    def animate(self, num_frames, final_x):

        # TODO: configurable?
        ylim = [-0.5, 1.5]

        # Create initial figure, axes, and ground line
        fig, ax = plt.subplots()
        xlim = [self.hip_x - 1, final_x + 1]
        ax.plot(xlim, [0, 0], "--", linewidth=5)

        actors = self.create_actors(ax)
        poses = self.generate_poses(num_frames, final_x)

        def init():
            """Initialize the animation axis."""
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            return actors

        def update(frame_index):
            """Update the animation axis with data from next frame."""
            self.update_actors(actors, poses, frame_index)
            return actors

        anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)  # type: ignore
        return anim


#%%


# body_length = 1
# body_width = 0.2

# upper_leg_length = 0.3
# lower_leg_length = 0.2
# Worry about hips rotating out later

foot_stride = 1.5
foot_lift = 0.2

initial_x = 1
final_x = 9

num_anim_frames = 32

animat = Animat(foot_stride, initial_x, foot_lift)
animation = animat.animate(num_anim_frames, final_x)
HTML(animation.to_jshtml())
# animation.to_html5_video()
# animation.save("two-legs.mp4")


# %%

from math import atan2, cos, degrees, pi, sin, sqrt


def simplify_angle(angle) -> float:
    angle = angle % (2.0 * pi)
    if angle < -pi:
        angle += 2.0 * pi
    elif angle > pi:
        angle -= 2.0 * pi
    return angle


class Pt:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y

    def __sub__(self, other) -> Pt:
        return Pt(self.x - other.x, self.y - other.y)

    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f})"

    def __repr__(self) -> str:
        return self.__str__()

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    @classmethod
    def angle_between(cls, pt1, pt2) -> float:
        pt1_angle = atan2(pt1.y, pt1.x)
        pt2_angle = atan2(pt2.y, pt2.x)
        return simplify_angle((pt2_angle - pt1_angle) % (2 * pi))


class HingedSegment:
    def __init__(
        self,
        global_angle: float,
        length: float,
        parent_or_location: Union[HingedSegment, Pt],
    ) -> None:

        # Angle is always with respect to the global x-y coordinate system
        self.ang = global_angle
        self.len = length

        # For type checking
        self.loc = Pt(0, 0)
        self.par: Optional[HingedSegment]
        self.chi: Optional[HingedSegment] = None

        if isinstance(parent_or_location, HingedSegment):
            self.par = parent_or_location
            self.update_from_parent(0)
        else:
            self.par = None
            self.loc = parent_or_location

    def __str__(self) -> str:
        return str(self.loc) + f" @ {degrees(self.ang): .3f}°"

    def __repr__(self) -> str:
        return self.__str__()

    def update_from_parent(self, delta_angle: float) -> None:
        self.ang += delta_angle
        self.loc.x = self.par.loc.x + self.par.len * cos(self.par.ang)
        self.loc.y = self.par.loc.y + self.par.len * sin(self.par.ang)
        if self.chi:
            self.chi.update_from_parent(delta_angle)

    def set_new_angle(self, new_angle: float) -> None:
        delta_angle = new_angle - self.ang
        self.ang = new_angle
        if self.chi:
            self.chi.update_from_parent(delta_angle)

    def get_tip_location(self) -> Pt:
        tip = Pt(0, 0)
        tip.x = self.loc.x + self.len * cos(self.ang)
        tip.y = self.loc.y + self.len * sin(self.ang)
        return tip


class SegmentChain:
    def __init__(
        self,
        base: Pt,
        num_segs: int,
        angles: Union[float, List[float]],
        lengths: Union[float, List[float]],
        save_state: Optional[bool] = False,
    ) -> None:
        # Expand angles if single number
        assert isinstance(angles, (float, int)) or len(angles) == num_segs
        angles = [angles] * num_segs if isinstance(angles, (float, int)) else angles

        # Expand angles if single number
        assert isinstance(lengths, (float, int)) or len(lengths) == num_segs
        lengths = [lengths] * num_segs if isinstance(lengths, (float, int)) else lengths

        # Create segments
        parent = base
        self.segments = []
        for i in range(num_segs):
            self.segments.append(HingedSegment(angles[i], lengths[i], parent))
            parent = self.segments[-1]

        # Connect to children
        for parent, child in zip(self.segments, self.segments[1:]):
            parent.chi = child

        self.effector = self.segments[-1].get_tip_location()

        self.save_state = save_state
        self.states = []
        if save_state:
            self.add_state()

    def run_steps(
        self, goal: Pt, num_steps: int
    ) -> Optional[List[List[Tuple[float, float]]]]:
        # TODO: add type
        for _ in range(num_steps):
            self.step_to_goal(goal)

        if self.save_state:
            return self.states

    def step_to_goal(self, goal: Pt) -> None:
        for seg in reversed(self.segments):
            to_effector = self.effector - seg.loc
            to_goal = goal - seg.loc

            new_angle = Pt.angle_between(to_effector, to_goal)
            seg.set_new_angle(new_angle + seg.ang)

            self.effector = self.segments[-1].get_tip_location()

            # TODO: Check for termination by comparing new x,y to goal
            # TODO: Check if still making progress
            # TODO: constraint to specific axis
            # TODO: add joint limits
            # TODO: add direction of effector
            # epsilon = 0.0001
            # trivial_arc_length = 0.00001

            if self.save_state:
                self.add_state()

    def add_state(self):
        step_states = []
        for seg in self.segments:
            step_states.append((seg.loc.x, seg.loc.y))
        step_states.append((self.effector.x, self.effector.y))
        self.states.append(step_states)

    def get_states(self):
        return self.states

    def plot(self, goal: Optional[Pt] = None) -> None:
        # All actuated segments
        for p1, p2 in zip(self.segments, self.segments[1:]):
            plt.plot([p1.loc.x, p2.loc.x], [p1.loc.y, p2.loc.y])

        # Last joint to the end
        plt.plot(
            [self.segments[-1].loc.x, self.effector.x],
            [self.segments[-1].loc.y, self.effector.y],
        )

        # Goal
        if goal:
            plt.plot(goal.x, goal.y, "o")

        plt.axis("equal")


#%%
base = Pt(0, 0)
goal = Pt(1.5, 0.8)

num_segs = 3
seg_angle = -pi / 2
# seg_angle = (0, -pi/2, -pi/2)
seg_length = 1
chain = SegmentChain(Pt(0, 3), num_segs, seg_angle, seg_length, True)

num_steps = 3
step_data = chain.run_steps(goal, num_steps)


ylim = [-0.5, 3.5]
xlim = [-0.5, 3.5]

# Create initial figure, axes, and ground line
fig, ax = plt.subplots()
ax.plot(xlim, [0, 0], "--", linewidth=5)
plt.plot(goal.x, goal.y, "o", markersize=10)

actors = []
for _ in range(num_segs):
    (seg_ln,) = ax.plot([], [], marker="o", linewidth=5)
    actors.append(seg_ln)


def init():
    """Initialize the animation axis."""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return actors


def update(frame_index):
    """Update the animation axis with data from next frame."""
    frame_data = step_data[frame_index]
    for actor, (x1, y1), (x2, y2) in zip(actors, frame_data, frame_data[1:]):
        actor.set_data([x1, x2], [y1, y2])
    return actors


anim = FuncAnimation(fig, update, frames=len(step_data), init_func=init, blit=True)  # type: ignore
HTML(anim.to_jshtml())
# anim.save("first-ik.gif")

# %%
