"""
Purpose: create a streamlit app to plot movement in real time. Later, will also accept inputs for new run
"""
from abc import update_abstractmethods
from os import wait, chdir
from re import L
from typing import List, Tuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

from animat import Animat
from point import Pt

# from controller import Actors

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

        self.num_actors = int(
            st.number_input(
                "Number of Legs", min_value=1, max_value=10, value=4, step=1
            )
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

        # TODO: not hardcoded
        self.animat = Animat(4, 4, 4, 4)
        self.animat.add_goal(6)

        self.create_actors()

        if self.run_btn:
            self.run()

    def run(self):
        self.plot.pyplot(plt)

        self.new_anim()

        # TODO: go until done
        for i in range(10):

            self.animat.move()
            updates = self.actorupdate_from_stepdata(
                self.animat.get_last_step_data(), self.actors
            )
            self.animate(updates)
            time.sleep(1)

        # for i in range(10):
        #     updates = self.get_poses(self.actors)
        #     self.animate(updates)
        #     # time.sleep(3)

    def new_anim(self):
        for actor in self.actors:
            actor.set_data([0, 0, 0, 0], [0, 1, 2, 3])

        # hip
        self.actors[-1].set_data(
            [self.animat.front_hip.pos.x, self.animat.back_hip.pos.x],
            [self.animat.front_hip.pos.y, self.animat.back_hip.pos.y],
        )

        self.plot.pyplot(plt)

    def animate(self, updates):
        for actor, x, y in updates:
            actor.set_data(x, y)

        # hip
        self.actors[-1].set_data(
            [self.animat.front_hip.pos.x, self.animat.back_hip.pos.x],
            [self.animat.front_hip.pos.y, self.animat.back_hip.pos.y],
        )
        self.plot.pyplot(plt)

    def create_actors(self):
        for i in range(0, self.num_actors):
            line, = self.ax.plot([], [], marker="o", linewidth=self.line_width)
            self.actors.append(line)

        # hip actor
        # line, = self.ax.plot([], [], marker="o", linewidth=self.line_width)
        # self.actors.append(line)

    def actorupdate_from_stepdata(
        self, data: List[List[Pt]], actors: List[Actor]
    ) -> List[Actor_Update]:
        # unzip x,y vals and assign to actors

        updates = []

        for i, leg in enumerate(data):

            x = []
            y = []

            for pt in leg:
                x.append(pt.x)
                y.append(pt.y)

            updates.append((actors[i], x, y))

        return updates

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

            updates.append((actor, x, y))

        return updates


app = App()
