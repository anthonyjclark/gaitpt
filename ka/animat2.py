from __future__ import annotations
from ka.hinged_segment import HingedSegment  # stops the errors, remove later
from ka.point import Pt
from ka.leg import Leg
from ka.gaits import Gait, FootState

from typing import List, Tuple, Optional, Union
from icecream import ic


from math import atan2, cos, degrees, pi, sin, sqrt


class Animat(object):
    def __init__(self, actors, num_segs, height):
        self.num_legs = len(actors)
        self.num_segs = num_segs
        self.ht = height

