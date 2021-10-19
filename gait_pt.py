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
    '''
    Enumerates FootStates using integers 1 - 5.

    GROUND (5) is being deprecated in favor of (TODO)?
    '''
    SWING = 1
    LIFT = 2
    THRUST = 3
    SUPPORT = 4

    # TODO: remove this one
    GROUND = 5


# Start with just one leg
def walk_footfalls():
    '''
    TODO: define walk pattern as sequence of states for each leg   
    '''
    ...


# %%


class Leg(object):
    def __init__(
        self, stride: float, state: FootState, foot_x: float, swing_y: float
    ) -> None:
        """
        Given stride length, foot state, and foot position (x and y), creates a 
        leg object.

        Args:
            stride (float): length of full stride
            state (FootState): FootState IntEnum; 1-4 supported
            foot_x (float): horizontal position of foot
            swing_y (float): vertical position of foot, if swinging
        """    

        super().__init__()
        self.state = state
        self.stride = stride
        self.half_stride = stride / 2
        self.foot_x = foot_x
        self.swing_y = swing_y

    def is_swinging(self) -> bool:
        '''
        Returns true if foot state is currently 'SWING'
        '''
        return self.state == FootState.SWING

    def not_swinging(self) -> bool:
        '''
        Returns true if not is_swinging()
        '''
        return not self.is_swinging()

    def motion_step(self, hip_x: float, foot_delta: float = 8 / (32/3)) -> Tuple[float, float]:
        """
        Given hip position (x) and using is_swinging results, set the new footstate
        or continue moving the foot, depending on whether a stride is completed.

        Args:
            hip_x (float): Position of hip
            foot_delta (float, optional): Amount to move foot. Added as parameter
            per todo comment. Defaults to 8/(32/3), which was original amount in 
            previous version of code.

        Returns:
            Tuple[float, float]: 
                - self.foot_x = current horizontal position of foot
                - self.swing_y = current height of foot
                - 0 if not self.is_swinging(): no change in position anticipated
        
        """        
       
        if self.not_swinging() and self.foot_x < hip_x - self.half_stride:
            # start swinging the foot
            self.state = FootState.SWING
        elif self.not_swinging():
            #not swinging and doesn't need to
            pass
        elif self.is_swinging() and self.foot_x > hip_x + self.half_stride:
            #completed the stride
            self.state = FootState.GROUND
        else:
            # if moving and not completed the stride, move a little bit more
            self.foot_x += foot_delta

        #TODO: should this return 0,0 if not swinging?
        return self.foot_x, self.swing_y if self.is_swinging() else 0

    def create_actors(self, ax: plt.Axes) -> Actors:
        """
        TODO: I don't really understand this section..

        Args:
            ax (plt.Axes): [description]

        Returns:
            Actors: [description]
        """        
        
        # TODO: linewidth as parameter
        (leg_ln,) = ax.plot([], [], marker="o", linewidth=5) #creates empty line bounded by "o" character
        return [leg_ln]


class Animat(object):
    # TODO: four legs, give center of body instead of hip
    # - hip height
    def __init__(self, stride: float, initial_x: float, foot_lift: float) -> None:
        """
        Initiates Animat by setting its hip position and adding legs in.

        Args:
            stride (float): length of a stride
            initial_x (float): initial position of object's hip
            foot_lift (float): vertical distance from ground when swinging (swing_y in Leg)
        """        
        super().__init__()

        self.hip_x = initial_x

        x = initial_x  # + stride / 2
        self.legs = [
            Leg(stride, FootState.GROUND, x, foot_lift),  # Back left
            Leg(stride, FootState.SWING, x, foot_lift),  # Back right
        ]
        #Question: how do we account for more legs? parameter? constant set at 2?

    def motion_step(self, new_x: float) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Given a final destination, calls motion step on each component leg,
        collecting information on each leg's position.

        Args:
            new_x (float): new position where movement ends

        Returns:
            Tuple[float, List[Tuple[float, float]]]: 
                - float: updates self.hip_x and returns it
                - List: array of leg positions
        """
        # TODO: change to body
        self.hip_x = new_x

        # TODO: how to collect pose info for multiple legs and multiple joints? (dict?)
        leg_poses = [leg.motion_step(self.hip_x) for leg in self.legs]

        return self.hip_x, leg_poses

    def generate_poses(self, num_steps: int, final_x: float) -> Poses:
        """
        Given a final position and the number of steps to get there, compiles
        a list of each position needed to get to final destination. 

        Args:
            num_steps (int): number of steps needed to get to final destination
            final_x (float): final destination

        Returns:
            Poses: result of calling motion_step on animat, which in turn calls motion_step
            for each leg. Lists positions of all legs at every step.
        """        
        xs = np.linspace(self.hip_x, final_x, num_steps) #breaks down distance depending on num_steps
        poses = [self.motion_step(x) for x in xs]
        return poses

    def create_actors(self, ax: plt.Axes) -> Actors:
        """
        TODO: I don't understand the function we extend, so don't understand this one

        Args:
            ax (plt.Axes): [description]

        Returns:
            Actors: [description]
        """        
        # TODO: create body actors

        actors = []
        for leg in self.legs:
            actors.extend(leg.create_actors(ax))

        return actors

    def update_actors(self, actors: Actors, poses: Poses, frame_index: int) -> None:
        """
        Given a list of poses and a frame index, sets the leg positions to appropriate values
        for that frame.

        Args:
            actors (Actors): [description]
            poses (Poses): list of all poses in all frames, including hip and foot positions in (hip, [foot_pos]) format
            frame_index (int): one specific frame that is being used, from poses
        """        
        hip_x, foot_poses = poses[frame_index]
        for leg_ln, (foot_x, foot_y) in zip(actors, foot_poses):
            leg_ln.set_data([hip_x, foot_x], [1, foot_y])

    def animate(self, num_frames, final_x):
        """
        TODO

        Args:
            num_frames ([type]): [description]
            final_x ([type]): [description]

        Returns:
            [type]: [description]
        """
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

#Section 2: Inverse Kinematics

from math import atan2, cos, degrees, pi, sin, sqrt


def simplify_angle(angle: float) -> float:
    """
    Simplifies by reducing the size of the angle. 
    Unsure about the logic here?
    TODO: change name to better reflect

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


class Pt:
    def __init__(self, x: float, y: float) -> None:
        '''
        Sets the x and y positions based on parameters
        '''
        self.x, self.y = x, y

    def __sub__(self, other) -> Pt:
        '''
        subtracts another point from current, returns tuple = 
        (diff_x, diff_y)
        '''
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

    #equiv to "static" in C or Java
    @classmethod
    def angle_between(cls: Pt, pt1: Pt, pt2: Pt) -> float:
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
        return simplify_angle((pt2_angle - pt1_angle) % (2 * pi))


class HingedSegment:
    def __init__(
        self,
        global_angle: float,
        length: float,
        parent_or_location: Union[HingedSegment, Pt],
    ) -> None:
        """takes a global angle, decides if it has a parent, then creates a HS object

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

        if isinstance(parent_or_location, HingedSegment):
            #if what was passed in is a HS, it is a parent
            self.par = parent_or_location
            self.update_from_parent(0)
        else:
            #here, the parent_or_location var doesn't have type HS, so it can't be a parent.
            #  it contains a location
            self.par = None
            self.loc = parent_or_location

    def __str__(self) -> str:
        """Represents object as string containing info on location and angle

        Returns:
            str: str(self.loc) + f" @ {degrees(self.ang): .3f}°"
        """        
        return str(self.loc) + f" @ {degrees(self.ang): .3f}°"

    def __repr__(self) -> str:
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
            #update child if present
            self.chi.update_from_parent(delta_angle)

    def set_new_angle(self, new_angle: float) -> None:
        """change self angle to parameter and propagate that change through children

        Args:
            new_angle (float): new angle for this segment
        """        
        delta_angle = new_angle - self.ang # the difference between new and old angles
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


class SegmentChain:
    def __init__(
        self,
        base: Pt,
        num_segs: int,
        angles: Union[float, List[float]],
        lengths: Union[float, List[float]],
        save_state: Optional[bool] = False,
    ) -> None:
        """chain of HingedSegments

        Args:
            base (Pt): location of hip
            num_segs (int): number of HS's present
            angles (Union[float, List[float]]): list of angles for each segment. Can be one representative angle or a list 
                eq in length to num_segs
            lengths (Union[float, List[float]]): list of lengths of each segment
            save_state (Optional[bool], optional): Specifies whether to mark down positional states for each movement. Defaults to False.
        """    
        # Expand angles if single number
        assert isinstance(angles, (float, int)) or len(angles) == num_segs
        angles = [angles] * num_segs if isinstance(angles, (float, int)) else angles

        # Expand lengths if single number
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
        self.states = [] #states may be added to later with add_state
        if save_state:
            self.add_state()

    def run_steps(
        self, goal: Pt, num_steps: int
    ) -> Optional[List[List[Tuple[float, float]]]]:
        """runs step_to_goal for each step in num_steps

        Args:
            goal (Pt): point to reach in this run
            num_steps (int): num steps required to get to goal

        Returns:
            Optional[List[List[Tuple[float, float]]]]: optionally returns self.states if self.save_states is set
        """    
        for _ in range(num_steps):
            self.step_to_goal(goal)

        if self.save_state:
            return self.states

    def step_to_goal(self, goal: Pt) -> None:
        """adjusts the position of all segments to get one step closer to the goal

        Args:
            goal (Pt): final point to reach, may not be reached this iteration
        """        
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
        """add a state for each segment containing info about their x,y coords and append that sublist 
            to the main self.states value
        """        
        step_states = []
        for seg in self.segments:
            step_states.append((seg.loc.x, seg.loc.y))
        step_states.append((self.effector.x, self.effector.y)) #effector = tip of last segment in chain
        self.states.append(step_states)

    def get_states(self) -> List[List[Tuple[float, float]]]:
        """self.states contains list of all states we've collected so far. this returns them all

        Returns:
            List[List[Tuple[float, float]]]: all states we've captured so far
        """        
        return self.states

    def plot(self, goal: Optional[Pt] = None) -> None:
        """plots a single frame

        Args:
            goal (Optional[Pt], optional): goal point, optional if we want to plot it. Defaults to None.
        """        
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
seg_angle = -pi / 2 #next segments are vertically below
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
