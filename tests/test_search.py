# Copyright 2023, Patrick Riley, github: pfrstg

import numpy as np
import voxart

def test_search_design_random():
    rng = np.random.default_rng()
    def random_goal():
        return rng.choice(a=[0, 1], p=[0.7, 0.3], size=(7, 7))
    goal = voxart.Goal.from_arrays(
        random_goal(), random_goal(), random_goal())
    goal.add_frame()
    design = voxart.search_design_random(goal)
    np.testing.assert_array_equal(goal.goal(0), design.projection(0))
    np.testing.assert_array_equal(goal.goal(1), design.projection(1))
    np.testing.assert_array_equal(goal.goal(2), design.projection(2))
