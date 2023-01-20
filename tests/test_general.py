# Copyright 2023, Patrick Riley, github: pfrstg

import numpy as np
import os
import pytest

import voxart

def test_empty_design():
    design = voxart.Design.from_size(4)
    assert design.vox.shape == (4, 4, 4)

def test_design_figs():
    # Only verifies that the code runs, not anything about the output
    design = voxart.Design.from_size(4)
    design.projections_fig()
    design.slices_fig()

def test_design_equality():
    design_same0 = voxart.Design.from_size(3)
    design_same0.vox[0, 1, 2] = voxart.FILLED
    design_same1 = voxart.Design.from_size(3)
    design_same1.vox[0, 1, 2] = voxart.FILLED
    design_diff = voxart.Design.from_size(3)
    design_diff.vox[1, 1, 1] = voxart.FILLED

    assert design_same0 == design_same1
    assert not design_same0 != design_same1
    assert not design_same0 == design_diff
    assert design_same0 != design_diff

def test_design_diff_sizes_not_equal():
    design0 = voxart.Design(np.ones([3, 3, 3]))
    design1 = voxart.Design(np.ones([4, 4, 4]))

    assert design0 != design1

def test_design_hash():
    design_same0 = voxart.Design.from_size(3)
    design_same0.vox[0, 1, 2] = voxart.FILLED
    design_same1 = voxart.Design.from_size(3)
    design_same1.vox[0, 1, 2] = voxart.FILLED
    design_diff = voxart.Design.from_size(3)
    design_diff.vox[1, 1, 1] = voxart.FILLED

    assert hash(design_same0) == hash(design_same1)
    assert hash(design_same0) != hash(design_diff)

    s = {design_same0}
    assert len(s) == 1
    s.add(design_same1)
    assert len(s) == 1
    s.add(design_diff)
    assert len(s) == 2

def test_design_save_load(tmp_path):
    fn = os.path.join(tmp_path, "design.npy")
    design = voxart.Design.from_size(3)
    design.vox[1, 1, 1] = voxart.FILLED
    design.vox[2, 2, 2] = voxart.FILLED
    design.save_npy(fn)
    got = voxart.Design.from_npy(fn)
    assert got == design

def random_goal(size, rng):
    def one_view():
        return rng.choice(a=[0, 1], p=[0.7, 0.3], size=(size, size))
    goal = voxart.Goal.from_arrays(
        one_view(), one_view(), one_view())
    goal.add_frame()
    return goal

@pytest.mark.parametrize("size", [3, 5, 10, 15])
@pytest.mark.parametrize("seed", [100, 200, 300])
def test_design_find_removable_from_goal(size, seed):
    rng = np.random.default_rng(seed)
    design = random_goal(size, rng).create_base_design()
    assert np.all(design.find_removable() == design.find_removable_slow())

@pytest.mark.parametrize("size", [3, 5, 10, 15])
@pytest.mark.parametrize("seed", [100, 200, 300])
def test_design_find_removable(size, seed):
    rng = np.random.default_rng(seed)
    design = voxart.Design(rng.choice(a=[0, voxart.FILLED], p=[0.7, 0.3], size=(size, size, size)))
    assert np.all(design.find_removable() == design.find_removable_slow())

def test_goal_from_size():
    goal = voxart.Goal.from_size(4)
    assert goal.size == 4

def test_goal_from_arrays():
    goal = voxart.Goal.from_arrays(
        [[1, 1], [1, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]])
    assert goal.size == 2
    np.testing.assert_array_equal(goal.goal(0), [[1, 1], [1, 1]])
    np.testing.assert_array_equal(goal.goal(1), [[0, 0], [1, 1]])
    np.testing.assert_array_equal(goal.goal(2), [[1, 0], [0, 0]])

def test_goal_equality():
    arr0 = np.zeros([3, 3])
    arr0[0, 0] = voxart.FILLED
    arr1 = np.zeros([3, 3])
    arr1[1, 1] = voxart.FILLED
    arr2 = np.zeros([3, 3])
    arr2[2, 2] = voxart.FILLED
    goal_same0 = voxart.Goal.from_arrays(arr0, arr1, arr2)
    goal_same1 = voxart.Goal.from_arrays(arr0, arr1, arr2)
    # Note that different order is still different even though it is conceptually the same!
    goal_diff0 = voxart.Goal.from_arrays(arr2, arr1, arr0)
    goal_diff1 = voxart.Goal.from_arrays(arr2, arr2, arr2)

    assert goal_same0 == goal_same1
    assert not goal_same0 != goal_same1
    assert not goal_same0 == goal_diff0
    assert goal_same0 != goal_diff0
    assert not goal_same0 == goal_diff1
    assert goal_same0 != goal_diff1

def test_goal_diff_sizes_not_equal():
    assert voxart.Goal.from_size(3) == voxart.Goal.from_size(3)
    assert voxart.Goal.from_size(3) != voxart.Goal.from_size(4)

def test_goal_hash():
    arr0 = np.zeros([3, 3])
    arr0[0, 0] = voxart.FILLED
    arr1 = np.zeros([3, 3])
    arr1[1, 1] = voxart.FILLED
    arr2 = np.zeros([3, 3])
    arr2[2, 2] = voxart.FILLED
    goal_same0 = voxart.Goal.from_arrays(arr0, arr1, arr2)
    goal_same1 = voxart.Goal.from_arrays(arr0, arr1, arr2)
    goal_diff = voxart.Goal.from_arrays(arr2, arr2, arr2)

    assert hash(goal_same0) == hash(goal_same1)
    assert hash(goal_same0) != hash(goal_diff)

    s = {goal_same0}
    assert len(s) == 1
    s.add(goal_same1)
    assert len(s) == 1
    s.add(goal_diff)
    assert len(s) == 2

def test_goal_add_frame():
    goal = voxart.Goal.from_size(4)
    assert goal.goal(0).sum() == 0
    assert goal.goal(1).sum() == 0
    assert goal.goal(2).sum() == 0
    goal.add_frame()
    assert goal.goal(0).sum() == 24
    assert goal.goal(1).sum() == 24
    assert goal.goal(2).sum() == 24
    np.testing.assert_array_equal(goal.goal(0),
                                  [[2, 2, 2, 2],
                                   [2, 0, 0, 2],
                                   [2, 0, 0, 2],
                                   [2, 2, 2, 2]])

def test_goal_fig():
    # Only verifies that the code runs, not anything about the output
    goal = voxart.Goal.from_size(4)
    goal.add_frame()
    goal.fig()

def test_goal_alternate_forms_no_symmetries_no_flips():
    goal = voxart.Goal.from_arrays(
        [[2, 0],
         [0, 0]],
        [[2, 2],
         [0, 0]],
        [[2, 2],
         [0, 2]])
    forms = list(goal.alternate_forms(include_flips=False))
    assert len(forms) == 16
    # Goal 0 is always the same
    for g in forms:
        np.testing.assert_array_equal(g.goal(0),
                                      [[2, 0],
                                       [0, 0]])
    # Goal 1 comes in blocks of 4
    for g in forms[0:4]:
        np.testing.assert_array_equal(g.goal(1),
                                      [[2, 2],
                                       [0, 0]])
    for g in forms[4:8]:
        np.testing.assert_array_equal(g.goal(1),
                                      [[2, 0],
                                       [2, 0]])
    for g in forms[8:12]:
        np.testing.assert_array_equal(g.goal(1),
                                      [[0, 0],
                                       [2, 2]])
    for g in forms[12:16]:
        np.testing.assert_array_equal(g.goal(1),
                                      [[0, 2],
                                       [0, 2]])
    # Goal 2 switches every time
    for g in forms[0:16:4]:
        np.testing.assert_array_equal(g.goal(2),
                                      [[2, 2],
                                       [0, 2]])
    for g in forms[1:16:4]:
        np.testing.assert_array_equal(g.goal(2),
                                      [[2, 2],
                                       [2, 0]])
    for g in forms[2:16:4]:
        np.testing.assert_array_equal(g.goal(2),
                                      [[2, 0],
                                       [2, 2]])
    for g in forms[3:16:4]:
        np.testing.assert_array_equal(g.goal(2),
                                      [[0, 2],
                                       [2, 2]])

def test_goal_alternate_forms_many_symmetries():
    goal = voxart.Goal.from_arrays(
        [[2, 0],
         [0, 0]],
        [[2, 0],
         [0, 2]],
        [[2, 2],
         [2, 2]])
    forms = list(goal.alternate_forms())
    assert len(forms) == 2
    # Goal 0 is always the same
    for g in forms:
        np.testing.assert_array_equal(g.goal(0),
                                      [[2, 0],
                                       [0, 0]])
    # Goal 1 has two forms
    np.testing.assert_array_equal(forms[0].goal(1),
                                  [[2, 0],
                                   [0, 2]])
    np.testing.assert_array_equal(forms[1].goal(1),
                                  [[0, 2],
                                   [2, 0]])
    # Goal 2 is always the same
    for g in forms:
        np.testing.assert_array_equal(g.goal(2),
                                      [[2, 2],
                                       [2, 2]])

@pytest.mark.parametrize("flip_axis", [1, 2])
def test_goal_alternate_forms_flips(flip_axis):
    axis0 = [[2, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    symmetric = [[2, 0, 2],
                 [0, 0, 0],
                 [2, 0, 2]]
    flippable = [[2, 2, 2],
                 [2, 0, 0],
                 [0, 0, 0]]
    if flip_axis == 1:
        goal = voxart.Goal.from_arrays(axis0, flippable, symmetric)
    elif flip_axis == 2:
        goal = voxart.Goal.from_arrays(axis0, symmetric, flippable)
    else:
        raise ValueError()
    forms = list(goal.alternate_forms())
    assert len(forms) == 8
    # Goal 0 is always the same
    for g in forms:
        np.testing.assert_array_equal(g.goal(0), axis0)
    non_flip_axis = (flip_axis % 2) + 1
    # Non flipped axis is the same
    for g in forms:
        np.testing.assert_array_equal(g.goal(non_flip_axis), symmetric)
    # Flipped axis has 8 forms we enumerate
    np.testing.assert_array_equal(forms[0].goal(flip_axis),
                                  [[2, 2, 2],
                                   [2, 0, 0],
                                   [0, 0, 0]])
    np.testing.assert_array_equal(forms[1].goal(flip_axis),
                                  [[2, 0, 0],
                                   [2, 0, 0],
                                   [2, 2, 0]])
    np.testing.assert_array_equal(forms[2].goal(flip_axis),
                                  [[0, 0, 0],
                                   [0, 0, 2],
                                   [2, 2, 2]])
    np.testing.assert_array_equal(forms[3].goal(flip_axis),
                                  [[0, 2, 2],
                                   [0, 0, 2],
                                   [0, 0, 2]])
    np.testing.assert_array_equal(forms[4].goal(flip_axis),
                                  [[0, 0, 0],
                                   [2, 0, 0],
                                   [2, 2, 2]])
    np.testing.assert_array_equal(forms[5].goal(flip_axis),
                                  [[0, 0, 2],
                                   [0, 0, 2],
                                   [0, 2, 2]])
    np.testing.assert_array_equal(forms[6].goal(flip_axis),
                                  [[2, 2, 2],
                                   [0, 0, 2],
                                   [0, 0, 0]])
    np.testing.assert_array_equal(forms[7].goal(flip_axis),
                                  [[2, 2, 0],
                                   [2, 0, 0],
                                   [2, 0, 0]])

def test_goal_create_base_design():
    xface = np.array(
        [[0, 2, 2],
         [2, 2, 2],
         [2, 2, 2]])
    yface = np.array(
        [[2, 2, 0],
         [2, 2, 0],
         [2, 2, 2]])
    zface = np.array(
        [[2, 2, 2],
         [2, 0, 2],
         [2, 2, 2]])
    goal = voxart.Goal.from_arrays(xface, yface, zface)
    design = goal.create_base_design()
    print(design.vox)
    # The x face removes 3 voxesl
    # The yface removes 6 voxels that don't intersect
    # The zface removes 3 voxels, one of which intersects the yface
    # The 2 * is because FILLED is 2
    assert design.vox.sum() == 2 * (27 - (3 + 6 + 3 - 1))
    np.testing.assert_array_equal(design.projection(0), xface)
    np.testing.assert_array_equal(design.projection(1), yface)
    np.testing.assert_array_equal(design.projection(2), zface)
