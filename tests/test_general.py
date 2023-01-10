# Copyright 2023, Patrick Riley, github: pfrstg

import numpy as np
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
    design_same0.vox[0, 1, 2] = 1
    design_same1 = voxart.Design.from_size(3)
    design_same1.vox[0, 1, 2] = 1
    design_diff = voxart.Design.from_size(3)
    design_diff.vox[1, 1, 1] = 1

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
    design_same0.vox[0, 1, 2] = 1
    design_same1 = voxart.Design.from_size(3)
    design_same1.vox[0, 1, 2] = 1
    design_diff = voxart.Design.from_size(3)
    design_diff.vox[1, 1, 1] = 1

    assert hash(design_same0) == hash(design_same1)
    assert hash(design_same0) != hash(design_diff)

    s = {design_same0}
    assert len(s) == 1
    s.add(design_same1)
    assert len(s) == 1
    s.add(design_diff)
    assert len(s) == 2

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
    arr0[0, 0] = 1
    arr1 = np.zeros([3, 3])
    arr1[1, 1] = 1
    arr2 = np.zeros([3, 3])
    arr2[2, 2] = 1
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
    arr0[0, 0] = 1
    arr1 = np.zeros([3, 3])
    arr1[1, 1] = 1
    arr2 = np.zeros([3, 3])
    arr2[2, 2] = 1
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
    assert goal.goal(0).sum() == 12
    assert goal.goal(1).sum() == 12
    assert goal.goal(2).sum() == 12
    np.testing.assert_array_equal(goal.goal(0),
                                  [[1, 1, 1, 1],
                                   [1, 0, 0, 1],
                                   [1, 0, 0, 1],
                                   [1, 1, 1, 1]])

def test_goal_fig():
    # Only verifies that the code runs, not anything about the output
    goal = voxart.Goal.from_size(4)
    goal.add_frame()
    goal.fig()

def test_goal_rotations_no_symmetries():
    goal = voxart.Goal.from_arrays(
        [[1, 0],
         [0, 0]],
        [[1, 1],
         [0, 0]],
        [[1, 1],
         [0, 1]])
    rotations = list(goal.rotations())
    assert len(rotations) == 16
    # Goal 0 is always the same
    for r in rotations:
        np.testing.assert_array_equal(r.goal(0),
                                      [[1, 0],
                                       [0, 0]])
    # Goal 1 comes in blocks of 4
    for r in rotations[0:4]:
        np.testing.assert_array_equal(r.goal(1),
                                      [[1, 1],
                                       [0, 0]])
    for r in rotations[4:8]:
        np.testing.assert_array_equal(r.goal(1),
                                      [[1, 0],
                                       [1, 0]])
    for r in rotations[8:12]:
        np.testing.assert_array_equal(r.goal(1),
                                      [[0, 0],
                                       [1, 1]])
    for r in rotations[12:16]:
        np.testing.assert_array_equal(r.goal(1),
                                      [[0, 1],
                                       [0, 1]])
    # Goal 2 switches every time
    for r in rotations[0:16:4]:
        np.testing.assert_array_equal(r.goal(2),
                                      [[1, 1],
                                       [0, 1]])
    for r in rotations[1:16:4]:
        np.testing.assert_array_equal(r.goal(2),
                                      [[1, 1],
                                       [1, 0]])
    for r in rotations[2:16:4]:
        np.testing.assert_array_equal(r.goal(2),
                                      [[1, 0],
                                       [1, 1]])
    for r in rotations[3:16:4]:
        np.testing.assert_array_equal(r.goal(2),
                                      [[0, 1],
                                       [1, 1]])

def test_goal_rotations_many_symmetries():
    goal = voxart.Goal.from_arrays(
        [[1, 0],
         [0, 0]],
        [[1, 0],
         [0, 1]],
        [[1, 1],
         [1, 1]])
    rotations = list(goal.rotations())
    assert len(rotations) == 2
    # Goal 0 is always the same
    for r in rotations:
        np.testing.assert_array_equal(r.goal(0),
                                      [[1, 0],
                                       [0, 0]])
    # Goal 1 has two forms
    np.testing.assert_array_equal(rotations[0].goal(1),
                                  [[1, 0],
                                   [0, 1]])
    np.testing.assert_array_equal(rotations[1].goal(1),
                                  [[0, 1],
                                   [1, 0]])
    # Goal 2 is always the same
    for r in rotations:
        np.testing.assert_array_equal(r.goal(2),
                                      [[1, 1],
                                       [1, 1]])

def test_goal_create_base_design():
    xface = np.array(
        [[0, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])
    yface = np.array(
        [[1, 1, 0],
         [1, 1, 0],
         [1, 1, 1]])
    zface = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]])
    goal = voxart.Goal.from_arrays(xface, yface, zface)
    design = goal.create_base_design()
    print(design.vox)
    # The x face removes 3 voxesl
    # The yface removes 6 voxels that don't intersect
    # The zface removes 3 voxels, one of which intersects the yface
    assert design.vox.sum() == 27 - (3 + 6 + 3 - 1)
    np.testing.assert_array_equal(design.projection(0), xface)
    np.testing.assert_array_equal(design.projection(1), yface)
    np.testing.assert_array_equal(design.projection(2), zface)
