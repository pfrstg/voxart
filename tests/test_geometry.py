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


def test_stl_runs(tmp_path, random_goal):
    results = voxart.search(
        random_goal, "random_face_first", num_iterations=20, top_n=1
    )
    _, design = results.best()[0]
    mesh = voxart.design_to_stl(design)
    mesh.save(os.path.join(tmp_path, "test.stl"))
