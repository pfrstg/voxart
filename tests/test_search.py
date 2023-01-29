# Copyright 2023, Patrick Riley, github: pfrstg

import numpy as np
import pytest

import voxart


def test_masks():
    masks = voxart.Masks(4)
    np.testing.assert_array_equal(
        masks.interior,
        [
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
        ],
    )

    np.testing.assert_array_equal(
        masks.faces,
        [
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            [
                [False, True, True, False],
                [True, False, False, True],
                [True, False, False, True],
                [False, True, True, False],
            ],
            [
                [False, True, True, False],
                [True, False, False, True],
                [True, False, False, True],
                [False, True, True, False],
            ],
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
        ],
    )

    np.testing.assert_array_equal(
        masks.front_faces,
        [
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            [
                [False, True, True, False],
                [True, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ],
            [
                [False, True, True, False],
                [True, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ],
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
        ],
    )

    np.testing.assert_array_equal(
        masks.edges,
        [
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            [
                [True, False, False, True],
                [False, False, False, False],
                [False, False, False, False],
                [True, False, False, True],
            ],
            [
                [True, False, False, True],
                [False, False, False, False],
                [False, False, False, False],
                [True, False, False, True],
            ],
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
        ],
    )


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def random_goal(rng):
    def one_view():
        return rng.choice(a=[voxart.EMPTY, voxart.FILLED], p=[0.7, 0.3], size=(7, 7))

    goal = voxart.Goal.from_arrays(one_view(), one_view(), one_view())
    goal.add_frame()
    return goal


@pytest.mark.parametrize(
    "strategy", ["random", "random_face_first", "random_clear_front"]
)
def test_search_design_random(random_goal, strategy):
    num_alternate_forms = len(list(random_goal.alternate_forms()))
    results = voxart.search(random_goal, strategy, 2, 3)
    assert len(results.best()) == 3
    df = results.all_objective_values(["form_idx", "is_starting"])
    # It's 3 because we always add a result of the starting value
    assert len(df) == 3 * num_alternate_forms
    best_label, best_design = results.best()[0]
    # We're going to do a fairly weak check here because of the rotations and
    # flips the actual goal can change.
    # TODO: once we label the results we can revisit this test
    # np.testing.assert_array_equal(random_goal.goal(0), best.projection(0))
    # np.testing.assert_array_equal(random_goal.goal(1), best.projection(1))
    # np.testing.assert_array_equal(random_goal.goal(2), best.projection(2))
    assert random_goal.goal(0).sum() == best_design.projection(0).sum()
    assert random_goal.goal(1).sum() == best_design.projection(1).sum()
    assert random_goal.goal(2).sum() == best_design.projection(2).sum()


def test_objective_function():
    # This has 4 edges, 2 faces, 1 interior
    design = voxart.Design(
        [
            [
                [2, 2, 2],
                [2, 0, 2],
                [2, 2, 2],
            ],
            [
                [0, 0, 0],
                [2, 2, 2],
                [0, 0, 0],
            ],
            [
                [2, 2, 2],
                [2, 0, 2],
                [2, 2, 2],
            ],
        ]
    )

    masks = voxart.Masks(3)
    func0 = voxart.ObjectiveFunction(face_weight=10, interior_weight=1)
    with pytest.raises(ValueError):
        func0(design)
    func0.set_masks(masks)
    assert func0(design) == 21

    func1 = voxart.ObjectiveFunction(face_weight=1, interior_weight=10, masks=masks)
    func1(design) == 12


def test_objective_function_connector():
    # This has 2 face, 1 interior, 3 connectors
    design = voxart.Design(
        [
            [
                [0, 0, 0],
                [1, 2, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [1, 2, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [1, 2, 0],
                [0, 0, 0],
            ],
        ]
    )

    func = voxart.ObjectiveFunction(
        face_weight=100,
        interior_weight=10,
        connector_weight=1,
        masks=voxart.Masks(design),
    )
    assert func(design) == 213


def test_get_neighbors_middle():
    vox = [3, 6, 9]
    assert np.all(
        np.stack(list(voxart.get_neighbors(vox, 100)))
        == [
            [2, 6, 9],
            [4, 6, 9],
            [3, 5, 9],
            [3, 7, 9],
            [3, 6, 8],
            [3, 6, 10],
        ]
    )


def test_get_neighbors_zeroedge():
    vox = [0, 0, 0]
    assert np.all(
        np.stack(list(voxart.get_neighbors(vox, 10)))
        == [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )


def test_get_neighbors_maxedge():
    vox = [9, 9, 9]
    assert np.all(
        np.stack(list(voxart.get_neighbors(vox, 10)))
        == [
            [8, 9, 9],
            [9, 8, 9],
            [9, 9, 8],
        ]
    )


def test_get_neighbors_nearedge():
    vox = [1, 3, 4]
    assert np.all(
        np.stack(list(voxart.get_neighbors(vox, 5)))
        == [
            [0, 3, 4],
            [2, 3, 4],
            [1, 2, 4],
            [1, 4, 4],
            [1, 3, 3],
        ]
    )


@pytest.fixture
def empty_design_5():
    goal = voxart.Goal.from_size(5)
    goal.add_frame()
    return goal.create_base_design()


def test_get_shortest_path_to_targets_single(empty_design_5, rng):
    target, distance, path = voxart.get_shortest_path_to_targets(
        empty_design_5, voxart.Masks(empty_design_5), {(1, 1, 1)}, rng
    )
    assert target == (1, 1, 1)
    assert distance == 2
    assert len(path), 2


def test_get_shortest_path_to_targets_two(empty_design_5, rng):
    target, distance, path = voxart.get_shortest_path_to_targets(
        empty_design_5, voxart.Masks(empty_design_5), {(1, 1, 1), (2, 2, 2)}, rng
    )
    assert target == (1, 1, 1)
    assert distance == 2
    assert len(path), 2


def test_get_shortest_path_to_targets_zero_dist(empty_design_5, rng):
    target, distance, path = voxart.get_shortest_path_to_targets(
        empty_design_5, voxart.Masks(empty_design_5), {(2, 0, 0)}, rng
    )
    assert target == (2, 0, 0)
    assert distance == 0
    assert len(path) == 0


def test_add_path_as_connectors():
    design = voxart.Design.from_size(3)
    design.voxels[1, 0, 0] = voxart.FILLED
    design.voxels[1, 1, 1] = voxart.CONNECTOR

    assert design.num_filled() == 1
    assert design.num_connectors() == 1

    voxart.add_path_as_connectors(
        design,
        [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (1, 1, 2),
        ],
    )

    assert design.num_filled() == 1
    assert design.num_connectors() == 3

    assert design.voxels[1, 0, 0] == voxart.FILLED
    assert design.voxels[1, 1, 0] == voxart.CONNECTOR
    assert design.voxels[1, 1, 1] == voxart.CONNECTOR
    assert design.voxels[1, 1, 2] == voxart.CONNECTOR


def test_search_connectors(empty_design_5, rng):
    design = empty_design_5
    design.voxels[1, 1, 1] = voxart.FILLED
    design.voxels[2, 2, 2] = voxart.FILLED

    # 44 is the frame and 2 of the filled pieces
    assert design.num_filled() == 46
    assert design.num_connectors() == 0

    results = voxart.search_connectors(design, 5, 1, rng=rng)
    _, got_design = results.best()[0]

    assert got_design.num_filled() == 46
    assert got_design.num_connectors() == 3
