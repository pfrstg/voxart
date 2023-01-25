# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

import copy
import functools
import heapq
import itertools
import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

import voxart


class Masks:
    def __init__(self, size_or_design: Union[int, voxart.Design]):
        if isinstance(size_or_design, voxart.Design):
            size = size_or_design.size
        else:
            size = size_or_design
        self.interior = np.full((size, size, size), False)
        self.interior[1 : size - 1, 1 : size - 1, 1 : size - 1] = True

        self.edges = np.full((size, size, size), False)
        for axis, idx0, idx1 in itertools.product(
            range(3), [0, size - 1], [0, size - 1]
        ):
            indices = [idx0, idx1]
            indices.insert(axis, slice(None))
            self.edges[tuple(indices)] = True

        self.faces = np.full((size, size, size), True) & ~self.interior & ~self.edges


class ObjectiveFunction:
    """A lower-is-better objective value."""

    def __init__(
        self,
        face_weight: float = 2.5,
        interior_weight: float = 1.0,
        connector_weight: float = 0.1,
        masks: Optional[Masks] = None,
    ):
        """Creates ObjectiveFunction.

        If you do not provide masks at creation, you need to call set_masks before
        calling this.
        """
        self.face_weight = face_weight
        self.interior_weight = interior_weight
        self.connector_weight = connector_weight
        self.masks = masks

    def set_masks(self, masks: Masks):
        self.masks = masks

    def __call__(self, design: voxart.Design) -> float:
        if not self.masks:
            raise ValueError("Must set masks before calling ObjectiveFunction")

        return (
            self.face_weight * (design.voxels[self.masks.faces] == voxart.FILLED).sum()
            + self.interior_weight
            * (design.voxels[self.masks.interior] == voxart.FILLED).sum()
            + self.connector_weight * (design.voxels == voxart.CONNECTOR).sum()
        )


@dataclass(order=True)
class SearchResultsEntry:
    # Note that we are using lower is better objective values, but because
    # we use heapq which is also lower is better, we have to invert this value
    label: Tuple = field(compare=False)
    design: voxart.Design = field(compare=False)
    objective_value: float


class SearchResults:
    """Maintains state about the entire search process for a design."""

    def __init__(self, top_n: int, obj_func: ObjectiveFunction):
        self._top_n_to_keep = top_n
        self._best_results_heap = []
        self._all_objective_values = []
        self._obj_func = obj_func

    def add(self, label: Tuple, design: voxart.Design):
        """Adds a given design result

        label is a tuple with arbitrary values that will be returned later in best and
         all_objective_values

        """
        # See comment in Entry for why the -
        entry = SearchResultsEntry(label, design, -self._obj_func(design))
        if len(self._best_results_heap) < self._top_n_to_keep:
            heapq.heappush(self._best_results_heap, entry)
        else:
            heapq.heappushpop(self._best_results_heap, entry)
        self._all_objective_values.append((label, -entry.objective_value))

    def best(self) -> List[Tuple[Tuple, voxart.Design]]:
        return [
            (e.label, e.design)
            for e in sorted(self._best_results_heap, key=lambda e: -e.objective_value)
        ]

    def all_objective_values(self, label_names: List[str]) -> pd.DataFrame:
        return pd.DataFrame(
            ((*l, v) for (l, v) in self._all_objective_values),
            columns=label_names + ["objective_value"],
        )


def _random_search(
    design: voxart.Design, valid: np.typing.NDArray, rng: np.random.Generator
):
    """Performs a random search of removable pieces.

    valid is a boolean array where True means the piece should be considered
    """
    while True:
        removable = design.find_removable() & valid
        if np.sum(removable) == 0:
            break
        removable_indices = np.where(removable)
        spot_idx = rng.integers(len(removable_indices[0]))
        spot = (
            removable_indices[0][spot_idx],
            removable_indices[1][spot_idx],
            removable_indices[2][spot_idx],
        )
        # print(f"Removing {spot}")
        design.voxels[spot] = 0


def _search_random(design: voxart.Design, masks: Masks, rng: np.random.Generator):
    _random_search(design, ~masks.edges, rng)


def _search_random_face_first(
    design: voxart.Design, masks: Masks, rng: np.random.Generator
):
    _random_search(design, masks.faces, rng)
    _random_search(design, masks.interior, rng)


def search(
    goal: voxart.Goal,
    strategy: str,
    num_iterations: int,
    top_n: int,
    obj_func: Optional[ObjectiveFunction] = None,
    rng: Optional[np.random.Generator] = None,
) -> SearchResults:
    if strategy == "random":
        search_fn = _search_random
    elif strategy == "random_face_first":
        search_fn = _search_random_face_first
    else:
        raise ValueError(f"Strategy not known {strategy}")

    if rng is None:
        rng = np.random.default_rng()

    masks = Masks(goal.size)
    if obj_func is None:
        obj_func = ObjectiveFunction()
    obj_func.set_masks(masks)

    results = SearchResults(top_n, obj_func)

    for form_idx, goal_form in enumerate(goal.alternate_forms()):
        print(f"Starting goal form {form_idx}")
        starting_design = goal_form.create_base_design()
        results.add((form_idx, True), starting_design)
        for _ in range(num_iterations):
            design = copy.deepcopy(starting_design)
            search_fn(design, masks, rng)
            results.add((form_idx, False), design)

    return results


def get_neighbors(vox: np.typing.ArrayLike, size: int) -> Iterator[np.typing.NDArray]:
    vox = np.asarray(vox)
    if vox.shape != (3,):
        raise ValueError(f"Only suport 3D neightbors, got shape {vox.shape}")
    for axis, delta in itertools.product([0, 1, 2], [-1, 1]):
        newval = vox[axis] + delta
        if newval < 0 or newval >= size:
            continue
        neighbor = np.copy(vox)
        neighbor[axis] = newval
        yield neighbor


@dataclass(order=True)
class _PathEntry:
    vox: Tuple[int, int, int] = field(compare=False)
    parent: Optional[Tuple[int, int, int]] = field(compare=False)
    distance: int


def get_shortest_path_to_targets(
    design: voxart.Design,
    masks: Masks,
    targets: Set[Tuple[int, int, int]],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Tuple[int, int, int], int, List[Tuple[int, int, int]]]:
    """Given a set of targets, find the shortest path to the edge of the design.

    If there are multiple targets with shortest paths or multiple shortest paths,
    a random one will be returned.
    I don't know if it is uniformly random or not. Probably not.
    We use the rng to randomly pick an edge to start from.
    """
    if rng is None:
        rng = np.random.default_rng()

    edge_vox = np.where(masks.edges)
    starting_idx = rng.integers(len(edge_vox[0]))
    starting_point = (
        edge_vox[0][starting_idx],
        edge_vox[1][starting_idx],
        edge_vox[2][starting_idx],
    )

    frontier = [_PathEntry(starting_point, None, 0)]
    visited = {}
    while True:
        entry = heapq.heappop(frontier)

        if entry.vox in visited:
            continue

        if entry.vox in targets:
            # Yay, we found a shortest path!
            path = []
            path_parent = entry.parent
            while path_parent is not None and not masks.edges[path_parent]:
                path.append(path_parent)
                path_parent = visited[path_parent]
            return entry.vox, entry.distance, path

        # print(f"Visiting {entry}")
        visited[entry.vox] = entry.parent
        for neighbor in get_neighbors(entry.vox, design.size):
            neighbor = tuple(neighbor)
            dist = entry.distance
            if design.voxels[neighbor] == voxart.EMPTY:
                dist += 1
            next_entry = _PathEntry(neighbor, entry.vox, dist)
            # print(f"\tInserting {next_entry}")
            heapq.heappush(frontier, next_entry)


def add_path_as_connectors(design: voxart.Design, path: List[Tuple[int, int, int]]):
    for vox in path:
        if design.voxels[vox] == voxart.EMPTY:
            design.voxels[vox] = voxart.CONNECTOR


def search_connectors(
    starting_design: voxart.Design,
    num_iterations: int,
    top_n: int,
    obj_func: Optional[ObjectiveFunction] = None,
    rng: Optional[np.random.Generator] = None,
) -> SearchResults:
    if rng is None:
        rng = np.random.default_rng()

    masks = Masks(starting_design)
    if obj_func is None:
        obj_func = ObjectiveFunction()
    obj_func.set_masks(masks)

    results = SearchResults(top_n, obj_func)
    for iter_idx in range(num_iterations):
        design = copy.deepcopy(starting_design)

        pending_vox = set(
            (x, y, z)
            for x, y, z in zip(
                *np.where((design.voxels == voxart.FILLED) & ~masks.edges)
            )
        )
        while pending_vox:
            # TODO: Is convering to list all the time someting I shoudl avoid?
            if len(pending_vox) < 5:
                active_subset = pending_vox
            else:
                active_subset_idx = set(
                    np.random.choice(
                        len(pending_vox), size=len(pending_vox) // 2, replace=False
                    )
                )
                assert len(active_subset_idx) == len(pending_vox) // 2
                active_subset = {
                    v for i, v in enumerate(pending_vox) if i in active_subset_idx
                }
            target, distance, path = get_shortest_path_to_targets(
                design, masks, active_subset, rng
            )
            assert target in pending_vox
            add_path_as_connectors(design, path)
            pending_vox.remove(target)

        print(f"search_connectors: completed {iter_idx}")
        results.add((iter_idx, design.num_connectors()), design)

    return results
