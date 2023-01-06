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

@pytest.mark.parametrize("strategy", ["random", "random_face_first"])
def test_search_design_random(random_goal, strategy):
    results = voxart.search(random_goal, strategy, 10, 3)
    assert len(results.best()) == 3
    assert len(results.all_objective_values()) == 10
    best = results.best()[0]
    np.testing.assert_array_equal(random_goal.goal(0), best.projection(0))
    np.testing.assert_array_equal(random_goal.goal(1), best.projection(1))
    np.testing.assert_array_equal(random_goal.goal(2), best.projection(2))

def test_objective_value():
    # This has 4 edges, 2 faces, 1 interior
    design = voxart.Design([[[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]],
                            [[0, 0 ,0],
                             [1, 1, 1],
                             [0, 0, 0]],
                            [[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]]])

    masks = voxart.Masks(3)
    assert voxart.objective_value(design, masks, face_weight=10, interior_weight=1) == 21
    assert voxart.objective_value(design, masks, face_weight=1, interior_weight=10) == 12
