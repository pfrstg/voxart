# Copyright 2023, Patrick Riley, github: pfrstg

import numpy as np
import pytest
import voxart

def test_masks():
    masks = voxart.Masks(4)
    np.testing.assert_array_equal(
        masks.interior,
        [[[False, False, False, False],
          [False, False, False, False],
          [False, False, False, False],
          [False, False, False, False]],

         [[False, False, False, False],
          [False,  True,  True, False],
          [False,  True,  True, False],
          [False, False, False, False]],

         [[False, False, False, False],
          [False,  True,  True, False],
          [False,  True,  True, False],
          [False, False, False, False]],

         [[False, False, False, False],
          [False, False, False, False],
          [False, False, False, False],
          [False, False, False, False]]])

    np.testing.assert_array_equal(
        masks.faces,
        [[[False, False, False, False],
          [False,  True,  True, False],
          [False,  True,  True, False],
          [False, False, False, False]],

         [[False,  True,  True, False],
          [ True, False, False,  True],
          [ True, False, False,  True],
          [False,  True,  True, False]],

         [[False,  True,  True, False],
          [ True, False, False,  True],
          [ True, False, False,  True],
          [False,  True,  True, False]],

         [[False, False, False, False],
          [False,  True,  True, False],
          [False,  True,  True, False],
          [False, False, False, False]]])

    np.testing.assert_array_equal(
        masks.edges,
        [[[ True,  True,  True,  True],
          [ True, False, False,  True],
          [ True, False, False,  True],
          [ True,  True,  True,  True]],

         [[ True, False, False,  True],
          [False, False, False, False],
          [False, False, False, False],
          [ True, False, False,  True]],

         [[ True, False, False,  True],
          [False, False, False, False],
          [False, False, False, False],
          [ True, False, False,  True]],

         [[ True,  True,  True,  True],
          [ True, False, False,  True],
          [ True, False, False,  True],
          [ True,  True,  True,  True]]])

@pytest.fixture
def rng():
    return np.random.default_rng()

@pytest.fixture
def random_goal(rng):
    def one_view():
        return rng.choice(a=[0, 1], p=[0.7, 0.3], size=(7, 7))
    goal = voxart.Goal.from_arrays(
        one_view(), one_view(), one_view())
    goal.add_frame()
    return goal

def test_search_design_random(random_goal):
    design = voxart.search_design_random(random_goal)
    np.testing.assert_array_equal(random_goal.goal(0), design.projection(0))
    np.testing.assert_array_equal(random_goal.goal(1), design.projection(1))
    np.testing.assert_array_equal(random_goal.goal(2), design.projection(2))

def test_search_design_random_face_first(random_goal):
    design = voxart.search_design_random_face_first(random_goal)
    np.testing.assert_array_equal(random_goal.goal(0), design.projection(0))
    np.testing.assert_array_equal(random_goal.goal(1), design.projection(1))
    np.testing.assert_array_equal(random_goal.goal(2), design.projection(2))
