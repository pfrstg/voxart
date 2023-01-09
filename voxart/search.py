# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

from typing import Callable, Optional

import copy
from dataclasses import dataclass, field
import functools
import heapq
import itertools
import numpy as np

import voxart

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
        print("foo4")


@dataclass(order=True)
class SearchResultsEntry:
    # Note that we are using lower is better objective values, but because
    # we use heapq which is also lower is better, we have to invert this value
    design: voxart.Design = field(compare=False)
    objective_value: float

class SearchResults:
    """Maintains state about the entire search process for a design."""

    def __init__(self, top_n: int, value_fn: Callable[[voxart.Design], float]):
        self._top_n_to_keep = top_n
        self._best_results_heap = []
        self._all_objective_values = []
        self._value_fn = value_fn

    def add(self, design):
        # See comment in Entry for why the -
        entry = SearchResultsEntry(design, -self._value_fn(design))
        if len(self._best_results_heap) < self._top_n_to_keep:
            heapq.heappush(self._best_results_heap, entry)
        else:
            heapq.heappushpop(self._best_results_heap, entry)
        self._all_objective_values.append(-entry.objective_value)

    def best(self):
        return [e.design
                for e in sorted(self._best_results_heap, key=lambda e: -e.objective_value)]

    def all_objective_values(self):
        return self._all_objective_values


def _random_search(design: voxart.Design, valid: np.typing.NDArray, rng: np.random.Generator):
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


def objective_value(design: voxart.Design, masks: Masks,
                    face_weight:float = 2.5, interior_weight:float = 1.0) -> float:
    """Returns a lower-is-better objective value."""
    return (face_weight * design.vox[masks.faces].sum() +
            interior_weight * design.vox[masks.interior].sum())

def _search_random(design: voxart.Design, masks: Masks, rng: np.random.Generator):
    _random_search(design, ~masks.edges, rng)

def _search_random_face_first(
        design: voxart.Design, masks: Masks, rng: np.random.Generator):
    _random_search(design, masks.faces, rng)
    _random_search(design, masks.interior, rng)

def search(goal: voxart.Goal,
           strategy: str,
           num_iterations: int,
           top_n: int,
           rng: Optional[np.random.Generator] = None) -> SearchResults:
    print("bar1")
    if strategy == "random":
        search_fn = _search_random
    elif strategy == "random_face_first":
        search_fn = _search_random_face_first
    else:
        raise ValueError(f"Strategy not known {strategy}")

    if rng is None:
        rng = np.random.default_rng()

    starting_design = goal.create_base_design()
    masks = Masks(goal.size)
    obj_fn = functools.partial(objective_value, masks=masks)

    results = SearchResults(top_n, obj_fn)

    for _ in range(num_iterations):
        design = copy.deepcopy(starting_design)
        search_fn(design, masks, rng)
        results.add(design)

    return results
