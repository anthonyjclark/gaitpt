from __future__ import annotations
from hinged_segment import HingedSegment  # stops the errors, remove later
from point import Pt
from leg import Leg
from gaits import Gait, FootState
from animat import Animat

from typing import List, Tuple, Optional, Union


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from math import atan2, cos, degrees, pi, sin, sqrt

import argparse

# from icecream import ic

from IPython.display import HTML

Actors = List[plt.Line2D]
Poses = List[Tuple[float, List[Tuple[float, float]]]]


def main(args):

    """
    Goal: be able to set different gaits and speeds, collect the angle data, and plot that information

    strategy: take command line arguments for gaits (int) and speeds (str?). plot, but also be able to turn the data
    into some sort of file (probably csv)
    """

    # take inputs for gait and speed. TODO: add speed

    # create animat based on arguments
    animat = Animat(args.range[0], args.xdelta, args.ydelta, 0)

    # set gait and goal for animat; run it to the goal and collect the end states
    animat.assign_gait(Gait[args.gait[0]])

    goal = Pt(args.range[1] - animat.get_length(), 0)
    animat.assign_gait(args.gait)

    # TODO: chart out what the states look like
    animat.move(goal)

    # animate those states after all collected
    states = animat.get_angles()

    animation = animate(goal.x, animat)

    HTML(animation.to_jshtml)
    print("did html stuff")

    # write states to a csv file

    return 0


def animate(final_x, animat: Animat, ylim_min=-0.5, ylim_max=10):
    """animates a number of frames by plotting them on a graph

    Args:
        num_frames ([type]): [description]
        final_x ([type]): [description]
        ylim_min (float, optional): [description]. Defaults to -0.5.
        ylim_max (float, optional): [description]. Defaults to 1.5.

    Returns:
        [type]: [description]
    """

    ylim = [ylim_min, ylim_max]

    # Create initial figure, axes, and ground line
    fig, ax = plt.subplots()
    xlim = [animat.get_pos() - animat.length, final_x + animat.length]
    ax.plot(xlim, [0, 0], "--", linewidth=5)

    leg_linewidth = 5

    actors = create_actors(animat.legs, ax, leg_linewidth)
    leg_states = animat.leg_angles

    def init():
        """Initialize the animation axis."""
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return actors

    def update(frame_index, actors, poses):
        """Update the animation axis with data from next frame."""
        update_actors(actors, poses, frame_index)
        return actors

    anim = FuncAnimation(fig, update, frames=len(leg_states[0]), init_func=init, blit=True, fargs=(actors, leg_states))  # type: ignore
    return anim


def create_actors(legs: list(Leg), ax, linewidth):
    actors = []

    for leg in legs:
        leg_actors = []

        for seg in leg.get_segments():
            (seg_ln,) = ax.plot([], [], marker="o", linewidth=linewidth)
            leg_actors.append(seg_ln)

        actors.append(leg_actors)

    return actors


def update_actors(actors: Actors, step_states, frame_idx):
    leg_poses = [
        step_states[0][frame_idx],
        step_states[1][frame_idx],
        step_states[2][frame_idx],
        step_states[3][frame_idx],
    ]  # TODO: add for loop to support more limbs

    # now foot poses and actors should line up perfectly
    for i, leg in enumerate(leg_poses):

        for actor, ([startx, starty], [endx, endy]) in zip(actors[i], leg):

            actor.set_data([startx, starty], [endx, endy])


parser = argparse.ArgumentParser(
    description="Enter information for gait and speeds requested"
)
parser.add_argument(
    "gait", type=str, nargs=1, help="One of: walk, pace, canter, gallop"
)
parser.add_argument(
    "--xdelta",
    type=float,
    nargs=1,
    default=1,
    help="horizontal speed of the animal",
)
parser.add_argument(
    "--ydelta",
    type=float,
    nargs=1,
    default=1,
    help="vertical speed of the animal",
)
parser.add_argument(
    "--segment_lens",
    type=float,
    nargs="+",
    default=[0.5, 0.5, 0.5, 0.5],
    help="list of lengths representing each section of the animals leg",
)
parser.add_argument(
    "--range",
    type=float,
    nargs=2,
    default=[0, 10],
    help="start and end of the plot",
)

args = parser.parse_args()

main(args)
