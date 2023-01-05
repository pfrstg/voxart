# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_masks(size) -> Tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    vox = np.zeros((size, size, size))
    interior = np.full((size, size, size), False)
    interior[1:size-1, 1:size-1, 1:size-1] = True

    edges = np.full((size, size, size), False)
    for axis, idx0, idx1 in itertools.product(range(3), [0, size-1], [0, size-1]):
        indices = [idx0, idx1]
        indices.insert(axis, slice(None))
        edges[tuple(indices)] = True

    faces = np.full((size, size, size), True) & ~interior & ~edges

    return interior, faces, edges


def search_design_random(goal, rng: Optional[np.random.Generator] = None) -> Design:
    if rng is None:
        rng = np.random.default_rng()

    starting_design = goal.create_base_design()
    interior, faces, edges = create_masks(goal.size)

    design = copy.deepcopy(starting_design)
    while True:
        removable = design.find_removable_slow() & ~edges
        if np.sum(removable) == 0:
            break
        removable_indices = np.where(removable)
        spot_idx = rng.integers(len(removable_indices[0]))
        spot = (removable_indices[0][spot_idx],
                removable_indices[1][spot_idx],
                removable_indices[2][spot_idx])
        #print(f"Removing {spot}")
        design.vox[spot] = 0
        assert np.all(design.projection(0) == starting_design.projection(0))
        assert np.all(design.projection(1) == starting_design.projection(1))
        assert np.all(design.projection(2) == starting_design.projection(2))

    return design
