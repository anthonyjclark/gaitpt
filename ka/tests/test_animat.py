"""
Need to test:
    - creating legs from app-created actors + some extra info that app collected (num joints, ht of animal)
    - assigning legs to 2 hips equally
    - taking steps to reach xlim as captured by app
        - setting sub-goals for each movement
        - reporting information to app for each movement
    - collecting all step info into one master list

"""
from typing import List
import pytest
import unittest
import numpy as np
from ka.leg import Leg
from ka.gaits import Gait, FootState
from ka.animat2 import Animat


class TestCreateLegs(unittest.TestCase):
    def setUp(self):
        self.wrong_actors = []
        self.correct_actors = []
        self.correct_num_actors = np.random.randint(1, 10)
        for i in range(0, self.correct_num_actors):
            line, = self.ax.plot([], [], marker="o", linewidth=self.line_width)
            self.correct_actors.append(line)

        self.wrong_height = 0
        self.correct_height = 4

        self.wrong_num_segs = 0
        self.correct_num_segs = 2

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

        # should return a list of Leg
        legs = Animat.make_legs(
            self.correct_actors, self.correct_height, self.correct_num_segs
        )

        assert isinstance(legs, List(Leg))

        # now check to make sure they're correct - env vironment vars have the right # of legs and segs to check against
        assert len(legs) == self.correct_num_actors

        # now check correct num segs for each leg
        for leg in legs:
            assert len(leg.segments) == self.correct_num_segs


class TestLegMovement(unittest.TestCase):
    def setUp(self):
        # need to initiate a complete Animat and give it a total goal
        self.goal = 10  # assume start at 0
        self.num_legs = 4
        self.num_segs = 2
        self.height = 4
        self.speed = 1
        self.x_delt = 1
        self.y_delt = 1

        self.gait = Gait.WALK

        self.animat = Animat(
            self.num_legs, self.num_segs, self.height
        )  # no ground anymore

    def testGiveGoal(self):
        """white-box
        GIVEN   a constructed animat and valid goal
        WHEN    Animat.set_goal() is called
        THEN    Animat will assign appropriate sub_goals to each leg, returning an array of poses. Will also 
                re-set each leg's index to 0
        
        """
        self.animat.set_speed(self.speed)  # need speed to calculate goal
        self.animat.set_goal(self.goal)

        subgoals = self.animat.get_subgoals()
        # check sg for 1) no movement > delta, correct # steps, up and down
        assert len(subgoals) == self.goal / self.speed

        last_dir = 0  # 0 = down, 1 = up
        last_s = (0, 0)
        for leg_goals in subgoals:
            for s in leg_goals:
                (x, y) = s
                if last_dir > 0:
                    assert y < 0
                    last = 0
                else:
                    assert y > 0
                    last = 1
                assert x <= last_s[0] + 1
                last_s = s

    def testCompleteSubgoal(self):
        """white-box
        GIVEN   a constructed animat and valid sub-goals
        WHEN    Animat.move() is called and the current sub_goal is complete
        THEN    Animat will assign the next sub_goal correctly to the completed leg
        
        """

        self.animat.set_speed(self.speed)  # need speed to calculate goal
        self.animat.set_goal(self.goal)

        for leg in self.animat.legs:
            assert leg.i == 0
            leg.move()
            assert leg.i == 1

    def testLegMovement(self):
        # maybe should go under leg tests
        """black-box
        GIVEN   a constructed animat and leg and valid sub-goals
        WHEN    Animat.move() is called
        THEN    Animat will make sure each leg moves to its goal and return the step info
        
        """

        self.animat.set_speed(self.speed)  # need speed to calculate goal
        self.animat.set_goal(self.goal)

        legs = self.animat.legs

        for leg in legs:
            leg.move()
            assert leg.effector == leg.goals[leg.i]  # got where it was supposed to go

            # get last step's data for the legs and compare that to the number of segments
            step_data_x, step_data_y = leg.get_last_step()
            assert len(step_data_x) == self.num_segs
            assert len(step_data_y) == self.num_segs
            assert len(step_data_x) == len(step_data_y)

    def testHipMovement(self):
        """white-box
        GIVEN   a constructed animat and leg and valid sub-goals
        WHEN    Animat.move() is called and a leg completes its sub-goal
        THEN    Animat will move the corresponding hip into the leg's new position
        
        """
        self.animat.set_speed(self.speed)  # need speed to calculate goal
        self.animat.set_goal(self.goal)

        legs = self.animat.legs

        for leg in legs:
            leg.move()
            hip_x, _ = leg.hip
            eff_x, _ = leg.effector
            assert hip_x == eff_x

    def testLegToActor(self):
        """white-box
        GIVEN   an Animat and constructed legs, plus app-constructed actors
        WHEN    Animat.to_actor() is called with an input of a single leg
        THEN    Animat will transform each leg into one actor and return a list of actors
        
        """
        # we need lists for x and y info from a leg
        for leg in self.animat.legs:
            xs, ys = leg.to_actor()
            assert len(xs) == len(ys)
            assert len(xs) == self.num_segs + 1

            assert (xs[0], ys[0]) == leg.hip
            for i in range(1, len(xs)):
                assert leg.segments[i].parent.loc == (xs[0], ys[0])

    def testAllStepsCollection(self):
        """white-box
        GIVEN   a constructed animat and leg and valid total goal
        WHEN    Animat.get_data() is called
        THEN    Animat will collect and return data on leg poses for all movement, writing it to a file
        
        """

        self.animat.set_speed(self.speed)  # need speed to calculate goal
        self.animat.set_goal(self.goal)
        self.animat.go()  # runs all the steps at once instead of one at a time

        data = self.animat.get_data()
        assert len(data) == self.num_legs

        for i, leg in enumerate(self.animat.legs):
            assert len(data[i]) == len(leg.get_goals())


"""black-box
    GIVEN   blah
    WHEN    blah
    THEN    blah
    
"""
if __name__ == "__main__":
    unittest.main()
