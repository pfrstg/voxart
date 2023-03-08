# Copyright 2023, Patrick Riley, github: pfrstg

import copy
import itertools
import os

import numpy as np
import pytest

import voxart


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


def test_stl_saving(tmp_path, random_goal):
    results = voxart.search_filled(
        random_goal, "random_face_first", num_iterations=20, top_n=1
    )
    _, design = results.best()[0]
    mesh = voxart.design_to_cube_stl(design, voxart.FILLED)
    mesh.save(os.path.join(tmp_path, "test.stl"))


def test_transform_stl_to_stand_on_point():
    design = voxart.Design.from_size(4)
    design.add_frame()
    orig_mesh = voxart.design_to_cube_stl(design, voxart.FILLED)
    for bottom_location in itertools.product([0, 1], repeat=3):
        mesh = copy.deepcopy(orig_mesh)
        voxart.transform_stl_to_stand_on_point(mesh, bottom_location, design.size)
        # This is not fully testing the functionality, just that for any bottom_location
        # we move all the points to be above the z axis which is a neccsary condition for a
        # good transform.
        assert np.all(mesh.vectors[:, :, 2] >= 0), bottom_location


@pytest.mark.parametrize(
    "connector_style,with_conn,bottom_location",
    [
        ("cube", True, None),
        ("cube", False, None),
        ("strut", True, None),
        ("strut", False, None),
        ("cube", True, [1, 1, 1]),
    ],
)
def test_save_model(tmp_path, random_goal, connector_style, with_conn, bottom_location):
    design = random_goal.create_base_design()
    design.add_frame()
    if with_conn:
        design.voxels[1, 1, 0] = voxart.CONNECTOR
    if bottom_location is not None:
        design.bottom_location = bottom_location
    stem = f"{tmp_path}/save_model"
    voxart.save_model_files(
        design,
        file_stem=stem,
        connector_style=connector_style,
    )
    assert os.path.exists(stem + "_filled.stl")
    if with_conn:
        assert os.path.exists(stem + "_connector.stl")
    else:
        assert not os.path.exists(stem + "_connector.stl")
    assert os.path.exists(stem + ".stl")
    assert os.path.exists(stem + ".3mf")
