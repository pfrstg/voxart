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


def test_masks_front_faces():
    masks = voxart.Masks(3)
    np.testing.assert_array_equal(
        masks.front_faces([0, 0, 0]),
        [
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ],
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
        ],
    )
    np.testing.assert_array_equal(
        masks.front_faces([-1, -1, -1]),
        [
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
            [
                [False, False, False],
                [False, False, True],
                [False, True, False],
            ],
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ],
        ],
    )


def test_masks_front_faces_invalid():
    masks = voxart.Masks(3)
    with pytest.raises(ValueError):
        masks.front_faces([2, 0, 0])
    with pytest.raises(ValueError):
        masks.front_faces([0, -2, 0])


def test_masks_single_edges():
    masks = voxart.Masks(3)
    single_edges = list(masks.single_edges())
    assert len(single_edges) == 12
    for edge in single_edges:
        assert np.sum(edge) == 3
    assert np.all(np.logical_or.reduce(single_edges) == masks.edges)


def test_masks_full_faces():
    masks = voxart.Masks(3)
    full_faces = list(masks.full_faces())
    assert len(full_faces) == 6
    for face in full_faces:
        assert np.sum(face) == 9
    assert np.all(np.logical_or.reduce(full_faces) == masks.edges | masks.faces)


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
    results = voxart.search_filled(random_goal, strategy, 2, 3)
    assert len(results.best()) == 3
    df = results.all_objective_values()
    # It's 3 because we always add a result of the starting value
    assert len(df) == 3 * num_alternate_forms
    assert list(df.columns) == [
        "form_idx",
        "is_starting",
        "iteration",
        "objective_value",
        "objective_value_rank",
    ]
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


def test_objective_function_unsupported():
    # This has 2 face, 1 interior, 3 connectors
    design = voxart.Design(
        [
            [
                [2, 2, 0],
                [2, 0, 0],
                [0, 0, 0],
            ],
            [
                [2, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 2],
            ],
        ]
    )

    func = voxart.ObjectiveFunction(
        face_weight=0,
        interior_weight=0,
        connector_weight=0,
        unsupported_weight=10,
        masks=voxart.Masks(design),
    )
    # no bottom, location, no connectors
    assert func(design) == 0
    design.bottom_location = (1, 1, 1)
    # still no connectors, so don't count
    assert func(design) == 0
    design.voxels[0, 0, 0] = voxart.CONNECTOR
    assert func(design) == 30
    design.bottom_location = None
    assert func(design) == 0


def test_objective_function_unsupported():
    design = voxart.Goal.from_size(5).create_base_design()
    func = voxart.ObjectiveFunction(
        face_weight=0,
        interior_weight=0,
        connector_weight=0,
        unsupported_weight=10,
        failure_penalty=10000,
        masks=voxart.Masks(design),
    )
    assert func(design) == 0
    design.failure = True
    assert func(design) == 10000


def test_get_neighbors_middle():
    vox = [3, 6, 9]
    assert np.all(
        np.fromiter(voxart.get_neighbors(vox, 100), dtype=(int, 3))
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
        np.fromiter(voxart.get_neighbors(vox, 10), dtype=(int, 3))
        == [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )


def test_get_neighbors_maxedge():
    vox = [9, 9, 9]
    assert np.all(
        np.fromiter(voxart.get_neighbors(vox, 10), dtype=(int, 3))
        == [
            [8, 9, 9],
            [9, 8, 9],
            [9, 9, 8],
        ]
    )


def test_get_neighbors_nearedge():
    vox = [1, 3, 4]
    assert np.all(
        np.fromiter(voxart.get_neighbors(vox, 5), dtype=(int, 3))
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
    masks = voxart.Masks(empty_design_5)
    target, distance, path = voxart.get_shortest_path_to_targets(
        empty_design_5,
        masks,
        list(zip(*np.where(masks.edges))),
        {(1, 1, 1)},
        allowed_mask=None,
        rng=rng,
    )
    assert target == (1, 1, 1)
    assert distance == 2
    assert len(path), 2


def test_get_shortest_path_to_targets_two(empty_design_5, rng):
    masks = voxart.Masks(empty_design_5)
    target, distance, path = voxart.get_shortest_path_to_targets(
        empty_design_5,
        masks,
        list(zip(*np.where(masks.edges))),
        {(1, 1, 1), (2, 2, 2)},
        allowed_mask=None,
        rng=rng,
    )
    assert target == (1, 1, 1)
    assert distance == 2
    assert len(path), 2


def test_get_shortest_path_to_targets_zero_dist(empty_design_5, rng):
    masks = voxart.Masks(empty_design_5)
    target, distance, path = voxart.get_shortest_path_to_targets(
        empty_design_5,
        masks,
        list(zip(*np.where(masks.edges))),
        {(2, 0, 0)},
        allowed_mask=None,
        rng=rng,
    )
    assert target == (2, 0, 0)
    assert distance == 0
    assert len(path) == 0


def test_get_shortest_path_to_targets_impossible(empty_design_5, rng):
    masks = voxart.Masks(empty_design_5)
    with pytest.raises(voxart.NoPathError):
        target, distance, path = voxart.get_shortest_path_to_targets(
            empty_design_5,
            masks,
            [(2, 2, 2)],
            {(2, 0, 0)},
            allowed_mask=masks.edges,
            rng=rng,
        )


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


def test_search_connectors_impossible(empty_design_5, rng):
    design = empty_design_5
    # Set all the back faces to obstruct pretty much everything
    design.voxels[-1, :, :] = voxart.FILLED
    design.voxels[:, -1, :] = voxart.FILLED
    design.voxels[:, :, -1] = voxart.FILLED
    # and then a voxel in the center that can't be connected
    design.voxels[2, 2, 2] = voxart.FILLED

    results = voxart.search_connectors(
        design, num_iterations=5, top_n=1, allow_obstructing=False, rng=rng
    )
    _, got_design = results.best()[0]

    assert got_design.failure


def test_connect_edges():
    design = voxart.Design.from_size(4)
    design.add_frame()

    design.voxels[1, 1, 1] = voxart.FILLED
    design.voxels[1, 1, 2] = voxart.FILLED

    # We need a constant rng because there are mutliple reasonable options
    # for shortest paths
    rng = np.random.default_rng(12345)
    voxart.connect_edges(design, voxart.Masks(design), rng)

    np.testing.assert_array_equal(
        design.voxels,
        [
            [
                [2, 2, 2, 2],
                [2, 0, 0, 2],
                [2, 0, 0, 2],
                [2, 2, 2, 2],
            ],
            [
                [2, 1, 0, 2],
                [1, 2, 2, 1],
                [0, 1, 0, 0],
                [2, 1, 1, 2],
            ],
            [
                [2, 1, 0, 2],
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [2, 1, 0, 2],
            ],
            [
                [2, 2, 2, 2],
                [2, 0, 0, 2],
                [2, 0, 0, 2],
                [2, 2, 2, 2],
            ],
        ],
    )


def test_connect_faces():
    design = voxart.Design.from_size(4)
    design.add_frame()

    design.voxels[1, 1, 1] = voxart.FILLED
    design.voxels[1, 1, 2] = voxart.FILLED

    # We need a constant rng because there are mutliple reasonable options
    # for shortest paths
    rng = np.random.default_rng(12345)
    voxart.connect_faces(design, rng=rng)

    np.testing.assert_array_equal(
        design.voxels,
        [
            [
                [2, 2, 2, 2],
                [2, 1, 0, 2],
                [2, 0, 0, 2],
                [2, 2, 2, 2],
            ],
            [
                [2, 1, 0, 2],
                [1, 2, 2, 1],
                [0, 0, 0, 0],
                [2, 0, 0, 2],
            ],
            [
                [2, 0, 0, 2],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [2, 1, 0, 2],
            ],
            [
                [2, 2, 2, 2],
                [2, 1, 0, 2],
                [2, 0, 0, 2],
                [2, 2, 2, 2],
            ],
        ],
    )


def test_is_vox_unsupported():
    design = voxart.Design.from_size(4)

    design.add_frame()
    design.bottom_location = (1, 1, 1)

    # With just a frame, everything on the edges is supported
    assert not voxart.is_vox_unsupported(design, (3, 3, 3))
    assert not voxart.is_vox_unsupported(design, (0, 0, 0))
    assert not voxart.is_vox_unsupported(design, (2, 0, 0))
    assert not voxart.is_vox_unsupported(design, (3, 3, 2))

    # No matter the bottom_location
    design.bottom_location = (1, 1, 1)
    assert not voxart.is_vox_unsupported(design, (3, 3, 3))
    assert not voxart.is_vox_unsupported(design, (0, 0, 0))
    assert not voxart.is_vox_unsupported(design, (2, 0, 0))
    assert not voxart.is_vox_unsupported(design, (3, 3, 2))

    # add one block that is suported in teh (0, 0, 0) direction
    design.voxels[1, 1, 1] = voxart.FILLED
    design.voxels[1, 0, 1] = voxart.CONNECTOR
    design.bottom_location = (0, 0, 0)
    assert not voxart.is_vox_unsupported(design, (1, 1, 1))
    design.bottom_location = (1, 1, 1)
    assert voxart.is_vox_unsupported(design, (1, 1, 1))
    design.bottom_location = (1, 0, 0)
    assert not voxart.is_vox_unsupported(design, (1, 1, 1))
    design.bottom_location = (0, 1, 0)
    assert voxart.is_vox_unsupported(design, (1, 1, 1))
    design.bottom_location = (0, 0, 1)
    assert not voxart.is_vox_unsupported(design, (1, 1, 1))


def test_count_unsupported():
    design = voxart.Design.from_size(4)

    design.add_frame()
    design.bottom_location = (0, 0, 0)
    assert voxart.count_unsupported(design) == 0

    design.voxels[1, 1, 1] = voxart.FILLED
    assert voxart.count_unsupported(design) == 1

    design.voxels[2, 2, 2] = voxart.CONNECTOR
    assert voxart.count_unsupported(design) == 2

    # Now add a support for the filled block
    design.voxels[0, 1, 1] = voxart.FILLED
    assert voxart.count_unsupported(design) == 1


def test_search_bottom_location():
    design = voxart.Design.from_size(4)

    design.add_frame()

    # add one block that is suported in teh (0, 0, 0) direction
    design.voxels[1, 1, 1] = voxart.FILLED
    design.voxels[1, 0, 1] = voxart.CONNECTOR
    obj_func = voxart.ObjectiveFunction(
        face_weight=0, interior_weight=0, connector_weight=0, unsupported_weight=1.0
    )

    results = voxart.search_bottom_location(design, obj_func)

    df = results.all_objective_values()
    assert np.all(
        df.loc[df["objective_value"] == 0.0, "bottom_location"]
        == ["(0, 0, 0)", "(0, 0, 1)", "(1, 0, 0)"]
    )
    assert np.sum(df["objective_value"]) == 5.0
