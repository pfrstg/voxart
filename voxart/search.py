# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Masks:
    def __init__(self, size: int):
        self.interior = np.full((size, size, size), False)
        self.interior[1:size-1, 1:size-1, 1:size-1] = True

        self.edges = np.full((size, size, size), False)
        for axis, idx0, idx1 in itertools.product(range(3), [0, size-1], [0, size-1]):
            indices = [idx0, idx1]
            indices.insert(axis, slice(None))
            self.edges[tuple(indices)] = True

        self.faces = np.full((size, size, size), True) & ~self.interior & ~self.edges


def _random_search(design: Design, valid: np.typing.NDArray, rng: np.random.Generator):
    """Performs a random search of removable pieces.

    valid is a boolean array where True means the piece shoud lbe considered
    """
    while True:
        removable = design.find_removable_slow() & valid
        if np.sum(removable) == 0:
            break
        removable_indices = np.where(removable)
        spot_idx = rng.integers(len(removable_indices[0]))
        spot = (removable_indices[0][spot_idx],
                removable_indices[1][spot_idx],
                removable_indices[2][spot_idx])
        #print(f"Removing {spot}")
        design.vox[spot] = 0


def search_design_random(goal: Goal, rng: Optional[np.random.Generator] = None) -> Design:
    if rng is None:
        rng = np.random.default_rng()

    starting_design = goal.create_base_design()
    masks = Masks(goal.size)

    design = copy.deepcopy(starting_design)
    _random_search(design, ~masks.edges, rng)

    return design

def search_design_random_face_first(
        goal: Goal, rng: Optional[np.random.Generator] = None) -> Design:
    if rng is None:
        rng = np.random.default_rng()

    starting_design = goal.create_base_design()
    masks = Masks(goal.size)

    design = copy.deepcopy(starting_design)
    _random_search(design, masks.faces, rng)
    _random_search(design, masks.interior, rng)

    print("hi")
    return design

#def objective_value(design
