# Copyright 2023, Patrick Riley, github: pfrstg

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
    results = voxart.search(
        random_goal, "random_face_first", num_iterations=20, top_n=1
    )
    _, design = results.best()[0]
    mesh = voxart.design_to_cube_stl(design, voxart.FILLED)
    mesh.save(os.path.join(tmp_path, "test.stl"))


@pytest.mark.parametrize(
    "connector_style,with_conn",
    [
        ("cube", True),
        ("cube", False),
        ("strut", True),
        ("strut", False),
    ],
)
def test_save_model(tmp_path, random_goal, connector_style, with_conn):
    design = random_goal.create_base_design()
    design.add_frame()
    if with_conn:
        design.voxels[1, 1, 0] = voxart.CONNECTOR
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
