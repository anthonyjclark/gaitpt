"""
Purpose: create a streamlit app to plot movement in real time. Later, will also accept inputs for new run
"""
from abc import update_abstractmethods
from re import L
from typing import List, Tuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time


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
        x_vals = np.arange(
            0, self.max_x
        )  # creates vector of evenly spaced values b/n 0 and max
        self.ax.set_ylim(0, self.max_y)
        self.ax.set_xlim(0, self.max_x)

        st.write(self.name)
        self.plot = st.pyplot(plt)

        self.num_actors = st.number_input(
            "Num Legs", min_value=1, max_value=10, value=4, step=1
        )

        for i in range(0, self.num_actors):
            line, = self.ax.plot([], [], marker="o", linewidth=self.line_width)
            self.actors.append(line)

    def run(self):

        actors = self.get_poses(
            self.actors
        )  # should start with a blank set of actors instead

        self.new_anim(actors)
        for i in range(100):
            updates = self.get_poses(self.actors)
            self.animate(updates)

    def new_anim(self, updates: Actor_Update):
        for actor, x, y in updates:
            actor.set_data(x, y)

    def animate(self, updates):
        for actor, x, y in updates:
            actor.set_data(x, y)
        self.plot.pyplot(plt)

    def get_poses(self, actors: List[Actor]) -> Actor_Update:
        # placeholder - this should be done by controller
        # gets new positions for each actor

        updates = []

        for i, actor in enumerate(actors):
            x = [i] + np.random.randint(i, self.max_y, self.max_x)
            y = np.random.randint(i, self.max_y / 2, self.max_x)
            # st.write(f"shape x = {len(x)}, shape y = {len(y)}")
            updates.append((actor, x, y))

        return updates


app = App()
app.run()
