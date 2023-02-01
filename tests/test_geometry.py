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
    mesh = voxart.design_to_stl(design, voxart.FILLED)
    mesh.save(os.path.join(tmp_path, "test.stl"))


def test_save_stl_pair(tmp_path, random_goal):
    design = random_goal.create_base_design()
    voxart.save_stl_pair(design, f"{tmp_path}/noconn")
    # import glob; print(glob.glob(f"{tmp_path}/*"))
    assert os.path.exists(f"{tmp_path}/noconn_filled.stl")
    assert not os.path.exists(f"{tmp_path}/noconn_connector.stl")
    design.voxels[1, 1, 1] = voxart.CONNECTOR
    voxart.save_stl_pair(design, f"{tmp_path}/withconn")
    # import glob; print(glob.glob(f"{tmp_path}/*"))
    assert os.path.exists(f"{tmp_path}/withconn_filled.stl")
    assert os.path.exists(f"{tmp_path}/withconn_connector.stl")
