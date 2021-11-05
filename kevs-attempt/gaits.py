from enum import Enum


class FootState(Enum):
    """classifies steps in terms of horizontal and vertical movement
    codifies a tuple describing (x_delta, y_delta)

    notes:
    - all steps are given the same base value. we can multiply it later if we
    want to increase speed

    """

    GROUND = 1  # no horizontal or vertical movement, except to get to ground as fast as possible
    SUSPEND = 2  # keeps horizontal movement, no vertical movement except to get air if not up already
    STEP = 3  # performs the full up and down movement of taking a step


class Gait(Enum):
    """Represents one of the gaits from https://www.animatornotebook.com/learn/quadrupeds-gaits"""

    # start by assuming only 4 legs, two hips
    # foot states are front to back, left to right
    WALK = [  # amble takes the same type of steps, will just be sped up
        (FootState.GROUND, FootState.GROUND, FootState.STEP, FootState.GROUND),
        (FootState.STEP, FootState.GROUND, FootState.GROUND, FootState.GROUND)(
            FootState.GROUND, FootState.GROUND, FootState.GROUND, FootState.STEP
        ),
        (FootState.GROUND, FootState.STEP, FootState.GROUND, FootState.GROUND),
    ]

    TROT = [
        (FootState.STEP, FootState.GROUND, FootState.GROUND, FootState.STEP),
        (FootState.GROUND, FootState.GROUND, FootState.GROUND, FootState.GROUND)(
            FootState.GROUND, FootState.STEP, FootState.STEP, FootState.NONE
        ),
        (FootState.GROUND, FootState.GROUND, FootState.GROUND, FootState.GROUND),
    ]

    PACE = [
        (FootState.STEP, FootState.GROUND, FootState.STEP, FootState.GROUND),
        (FootState.GROUND, FootState.GROUND, FootState.GROUND, FootState.GROUND)(
            FootState.GROUND, FootState.STEP, FootState.GROUND, FootState.STEP
        ),
        (FootState.GROUND, FootState.GROUND, FootState.GROUND, FootState.GROUND),
    ]

    # we're going to assume a left lead for gaits that have either left or right leads
    CANTER = [
        (FootState.GROUND, FootState.GROUND, FootState.STEP, FootState.GROUND),
        (FootState.GROUND, FootState.STEP, FootState.GROUND, FootState.STEP)(
            FootState.STEP, FootState.GROUND, FootState.GROUND, FootState.GROUND
        ),
        (FootState.SUSPEND, FootState.SUSPEND, FootState.SUSPEND, FootState.SUSPEND),
    ]

    GALLOP = [
        (FootState.GROUND, FootState.GROUND, FootState.STEP, FootState.GROUND),
        (FootState.GROUND, FootState.STEP, FootState.GROUND, FootState.STEP)(
            FootState.STEP, FootState.GROUND, FootState.GROUND, FootState.GROUND
        ),
        (FootState.GROUND, FootState.STEP, FootState.GROUND, FootState.GROUND),
        (FootState.SUSPEND, FootState.SUSPEND, FootState.SUSPEND, FootState.SUSPEND),
    ]
