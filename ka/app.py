"""
Purpose: create a streamlit app to plot movement in real time. Later, will also accept inputs for new run
"""
from abc import update_abstractmethods
from os import wait
from re import L
from typing import List, Tuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

from controller import Actors

Actor = Line2D
Actor_Update = Tuple[Actor, list[float], list[float]]  # an actor plus x and y values


class App:
    def __init__(self):
        self.name = "GaitPt Visualizer"
        self.max_x = 10
        self.max_y = 10
        self.line_width = 1
        self.actors = []

        self.fig, self.ax = plt.subplots()

        # set up axes
        self.ax.set_ylim(0, self.max_y)
        self.ax.set_xlim(0, self.max_x)

        st.write(self.name)

        self.plot = st.empty()
        self.run_btn = st.button(
            "Run", key="run", help="Set parameters first, then click here to run"
        )

        self.num_actors = st.number_input(
            "Number of Legs", min_value=1, max_value=10, value=4, step=1
        )

        self.num_joints = st.number_input(
            "Number of Joints Per Leg", min_value=1, max_value=10, value=2, step=1
        )

        self.anim_ht = st.number_input(
            "Height of Animal", min_value=1, max_value=10, value=4, step=1
        )

        self.max_x = st.number_input(
            "Distance To Travel", min_value=10, max_value=100, value=10, step=5
        )

        self.speed = st.number_input(
            "Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.1
        )

        self.gait = st.selectbox(
            "Choose a Gait", ("Walk", "Pace", "Canter", "Gallop"), 0, key="gait"
        )

        self.ax.set_xlim(0, self.max_x)

        self.create_actors()

        if self.run_btn:
            self.run()

    def run(self):
        self.plot.pyplot(plt)

        actors = self.get_poses(
            self.actors
        )  # should start with a blank set of actors instead

        self.new_anim(actors)
        for i in range(10):
            updates = self.get_poses(self.actors)
            self.animate(updates)
            time.sleep(3)

    def new_anim(self, updates: Actor_Update):
        for actor, x, y in updates:
            actor.set_data(x, y)

    def animate(self, updates):
        for actor, x, y in updates:
            actor.set_data(x, y)
        self.plot.pyplot(plt)

    def create_actors(self) -> Actors:
        for i in range(0, self.num_actors):
            line, = self.ax.plot([], [], marker="o", linewidth=self.line_width)
            self.actors.append(line)

    def get_poses(self, actors: List[Actor]) -> Actor_Update:
        # placeholder - this should be done by controller
        # gets new positions for each actor

        updates = []

        for i, actor in enumerate(actors):
            x = [i] + np.random.randint(0, self.max_x, size=self.num_joints)
            # TODO: will cause error when # legs > 7
            y = [int(self.anim_ht)] + np.random.randint(
                0, self.anim_ht, size=self.num_joints
            )

            # st.write(f"shape x = {len(x)}, shape y = {len(y)}")
            updates.append((actor, x, y))

        return updates


app = App()
