import numpy.testing as nptesting
import voxart

def test_empty_design():
    design = voxart.Design.from_size(4)
    assert design.vox.shape == (4, 4, 4)

def test_design_figs():
    # Only verifies that the code runs, not anything about the output
    design = voxart.Design.from_size(4)
    design.projections_fig()
    design.slices_fig()

def test_goal_from_size():
    goal = voxart.Goal.from_size(4)
    assert goal.size == 4

def test_goal_from_arr():
    goal = voxart.Goal.from_arrays(
        [[1, 1], [1, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]])
    assert goal.size == 2
    nptesting.assert_array_equal(goal.goal(0), [[1, 1], [1, 1]])
    nptesting.assert_array_equal(goal.goal(1), [[0, 0], [1, 1]])
    nptesting.assert_array_equal(goal.goal(2), [[1, 0], [0, 0]])

def test_goal_add_frame():
    goal = voxart.Goal.from_size(4)
    assert goal.goal(0).sum() == 0
    assert goal.goal(1).sum() == 0
    assert goal.goal(2).sum() == 0
    goal.add_frame()
    assert goal.goal(0).sum() == 12
    assert goal.goal(1).sum() == 12
    assert goal.goal(2).sum() == 12

def test_goal_fig():
    # Only verifies that the code runs, not anything about the output
    goal = voxart.Goal.from_size(4)
    goal.add_frame()
    goal.fig()

# TODO: test goal rotations, test goal base design
