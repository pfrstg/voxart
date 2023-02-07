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
    "connector_style,separate_files,expect_filled,expect_joint",
    [
        ("cube", True, True, False),
        ("cube", False, False, True),
        ("strut", True, True, False),
        ("strut", False, False, True),
    ],
)
def test_save_stl_noconn(
    tmp_path, random_goal, connector_style, separate_files, expect_filled, expect_joint
):
    design = random_goal.create_base_design()
    stem = f"{tmp_path}/save_stl_noconn"
    voxart.save_stl(
        design,
        file_stem=stem,
        connector_style=connector_style,
        separate_files=separate_files,
    )
    if expect_filled:
        assert os.path.exists(stem + "_filled.stl")
    else:
        assert not os.path.exists(stem + "_filled.stl")
    if expect_joint:
        assert os.path.exists(stem + ".stl")
    else:
        assert not os.path.exists(stem + ".stl")


@pytest.mark.parametrize(
    "connector_style,separate_files,expect_separate,expect_joint",
    [
        ("cube", True, True, False),
        ("cube", False, False, True),
        ("strut", True, True, False),
        ("strut", False, False, True),
    ],
)
def test_save_stl_conn(
    tmp_path,
    random_goal,
    connector_style,
    separate_files,
    expect_separate,
    expect_joint,
):
    design = random_goal.create_base_design()
    design.add_frame()
    design.voxels[1, 1, 0] = voxart.CONNECTOR
    stem = f"{tmp_path}/save_stl_noconn"
    voxart.save_stl(
        design,
        file_stem=stem,
        connector_style=connector_style,
        separate_files=separate_files,
    )
    if expect_separate:
        assert os.path.exists(stem + "_filled.stl")
        assert os.path.exists(stem + "_connector.stl")
    else:
        assert not os.path.exists(stem + "_filled.stl")
        assert not os.path.exists(stem + "_connector.stl")
    if expect_joint:
        assert os.path.exists(stem + ".stl")
    else:
        assert not os.path.exists(stem + ".stl")
