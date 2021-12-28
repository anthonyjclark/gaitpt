"""
Need to test:
    - creating legs from app-created actors + some extra info that app collected (num joints, ht of animal)
    - assigning legs to 2 hips equally
    - taking steps to reach xlim as captured by app
        - setting sub-goals for each movement
        - reporting information to app for each movement
    - collecting all step info into one master list

"""
import pytest
import unittest
import numpy as np
from animat import Animat


class TestCreateLegs(unittest.TestCase):
    def __init__(self):
        self.wrong_actors = []
        self.correct_actors = []
        for i in range(0, np.random.randint(1, 10)):
            line, = self.ax.plot([], [], marker="o", linewidth=self.line_width)
            self.correct_actors.append(line)

        self.wrong_height = 0
        self.correct_height = 4

        self.wrong_num_segs = 0
        self.correct_num_segs = 1

    def test_incorr_args(self):
        """white-box test
        GIVEN   0 or fewer actors, invalid height, or invalid num_segments passed
        WHEN    create_legs method is called
        THEN    descriptive error must be raised
        """

        # wrong actors, everything else fine
        self.assertRaises(
            ValueError,
            Animat.create_legs(
                self.wrong_actors, self.correct_height, self.correct_num_segs
            ),
        )

        # wrong height, everything else fine
        self.assertRaises(
            ValueError,
            Animat.create_legs(
                self.correct_actors, self.wrong_height, self.correct_num_segs
            ),
        )

        # wrong num segs, everything else fine
        self.assertRaises(
            ValueError,
            Animat.create_legs(
                self.correct_actors, self.correct_height, self.wrong_num_segs
            ),
        )

    def test_corr_legs(self):
        """black-box
        GIVEN   valid arguments
        WHEN    create_legs method is called
        THEN    should produce appropriate number of Leg objects and correct segments list for each
        """


"""black-box
    GIVEN   blah
    WHEN    blah
    THEN    blah
    
"""

