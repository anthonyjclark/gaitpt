from hinged_segment import HingedSegment  # stops the errors, remove later
from point import Pt
from leg import Leg
from gaits import Gait, FootState
from animat import Animat

from typing import List, Tuple, Optional, Union
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from math import atan2, cos, degrees, pi, sin, sqrt

import argparse

from icecream import ic

from IPython.display import HTML

Actors = List[plt.Line2D]
Poses = List[Tuple[float, List[Tuple[float, float]]]]


def main():

    """
    Goal: be able to set different gaits and speeds, collect the angle data, and plot that information

    strategy: take command line arguments for gaits (int) and speeds (str?). plot, but also be able to turn the data
    into some sort of file (probably csv)
    """

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
        type=List(float),
        nargs="+",
        default=[0.5, 0.5, 0.5, 0.5],
        help="list of lengths representing each section of the animals leg",
    )
    parser.add_argument(
        "--range",
        type=List(float),
        nargs=2,
        default=[0, 10],
        help="start and end of the plot",
    )

    args = parser.parse_args()

    # take inputs for gait and speed. TODO: add speed

    # create animat based on arguments
    animat = Animat(args.range[0], args.xdelta, args.ydelta, 0)

    # set gait and goal for animat; run it to the goal and collect the end states
    animat.assign_gait(args.gait)

    goal = Pt(args.range[1] - (animat.get_length))

    # TODO: chart out what the states look like

    # animate those states after all collected

    # write states to a csv file

    return 0


def animate(num_frames, final_x, ylim_min=-0.5, ylim_max=1.5):
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
    xlim = [self.hip_x - 1, final_x + 1]
    ax.plot(xlim, [0, 0], "--", linewidth=5)

    leg_linewidth = 5

    actors = self.create_actors(ax, leg_linewidth)
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


if "__name__" == main:
    main()
