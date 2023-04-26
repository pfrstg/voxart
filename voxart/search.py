# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

import copy
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange

import voxart


class Masks:
    """Useful True/False arrays for working with Designs.

    edges: the with 1 region at the exteriors
    faces: extenrior faces, not including the edges
    interior: not edges or faces
    single_edges: generator for each edge. Note tha the corners are included in
      3 single edges
    full_faces: generator for each face, including the edges on that face
    """

    def __init__(self, size_or_design: Union[int, voxart.Design]):
        if isinstance(size_or_design, voxart.Design):
            size = size_or_design.size
        else:
            size = size_or_design
        self.interior = np.full((size, size, size), False)
        self.interior[1 : size - 1, 1 : size - 1, 1 : size - 1] = True

        self.edges = np.full((size, size, size), False)
        for edge in self.single_edges():
            self.edges |= edge

        self.faces = np.full((size, size, size), True) & ~self.interior & ~self.edges

    @property
    def size(self) -> int:
        return self.interior.shape[0]

    def front_faces(self, goal_locations: List[int]) -> np.typing.NDArray:
        """Mask for the faces where the goals are frontwards.

        goal_locations woudl typically be from a Design object.
        """
        goal_locations = np.asarray(goal_locations)
        if not np.all((goal_locations == 0) | (goal_locations == -1)):
            raise ValueError(f"goal_locations must be 0,-1, got {goal_locations}")
        front_faces = np.full((self.size, self.size, self.size), False)
        front_faces[goal_locations[0], :, :] = True
        front_faces[:, goal_locations[1], :] = True
        front_faces[:, :, goal_locations[2]] = True
        front_faces &= self.faces
        return front_faces

    def single_edges(self):
        # Note that this method is called in init so be careful if it relies on anything else
        for axis, idx0, idx1 in itertools.product(
            range(3), [0, self.size - 1], [0, self.size - 1]
        ):
            edge = np.full((self.size, self.size, self.size), False)
            indices = [idx0, idx1]
            indices.insert(axis, slice(None))
            edge[tuple(indices)] = True
            yield edge

    def full_faces(self):
        for axis, idx in itertools.product(range(3), [0, -1]):
            full_face = np.full((self.size, self.size, self.size), False)
            indices = [slice(None), slice(None)]
            indices.insert(axis, idx)
            full_face[tuple(indices)] = True
            yield full_face


class ObjectiveFunction:
    """A lower-is-better objective value."""

    def __init__(
        self,
        face_weight: float = 3.9,
        interior_weight: float = 2.9,
        connector_weight: float = 1.0,
        unsupported_weight: float = 1.9,
        failure_penalty: float = 10000,
        masks: Optional[Masks] = None,
    ):
        """Creates ObjectiveFunction.

        If you do not provide masks at creation, you need to call set_masks before
        calling this.
        """
        self.face_weight = face_weight
        self.interior_weight = interior_weight
        self.connector_weight = connector_weight
        self.unsupported_weight = unsupported_weight
        self.failure_penalty = failure_penalty
        self.masks = masks

    def set_masks(self, masks: Masks):
        self.masks = masks

    def __call__(self, design: voxart.Design) -> float:
        if not self.masks:
            raise ValueError("Must set masks before calling ObjectiveFunction")

        value = (
            self.face_weight * (design.voxels[self.masks.faces] == voxart.FILLED).sum()
            + self.interior_weight
            * (design.voxels[self.masks.interior] == voxart.FILLED).sum()
            + self.connector_weight * (design.voxels == voxart.CONNECTOR).sum()
        )
        if design.bottom_location is not None and design.num_connectors():
            value += self.unsupported_weight * count_unsupported(design)
        if design.failure:
            value += self.failure_penalty
        return value


def data_ranks(a):
    """Computes ranks, dealing with ties by assigning lowest rank of group."""
    a = np.asarray(a)
    s = np.argsort(a)
    ranks = np.empty_like(s)
    ranks[s] = np.arange(len(a))
    ties = np.flatnonzero(np.diff(a[s]) == 0)
    # print(ties)
    for tie_idx in ties:
        ranks[s[tie_idx + 1]] = ranks[s[tie_idx]]
    return ranks


@dataclass(order=True)
class SearchResultsEntry:
    # Note that we are using lower is better objective values, but because
    # we use heapq which is also lower is better, we have to invert this value
    label: Tuple = field(compare=False)
    design: voxart.Design = field(compare=False)
    objective_value: float


class SearchResults:
    """Maintains state about the entire search process for a design."""

    def __init__(
        self, top_n: Optional[int], obj_func: ObjectiveFunction, label_names: List[str]
    ):
        self._top_n_to_keep = top_n
        self._best_results_heap = []
        self._all_objective_values = []
        self._obj_func = obj_func
        self._label_names = label_names

    @property
    def label_names(self) -> List[str]:
        return self._label_names

    def add(self, label: Tuple, design: voxart.Design):
        """Adds a given design result

        label is a tuple with arbitrary values that will be returned later in best and
         all_objective_values

        """
        # See comment in Entry for why the -
        entry = SearchResultsEntry(label, design, -self._obj_func(design))
        if (
            self._top_n_to_keep is None
            or len(self._best_results_heap) < self._top_n_to_keep
        ):
            heapq.heappush(self._best_results_heap, entry)
        else:
            heapq.heappushpop(self._best_results_heap, entry)
        self._all_objective_values.append((label, -entry.objective_value))

    def best(self) -> List[Tuple[Tuple, voxart.Design]]:
        return [
            (e.label, e.design)
            for e in sorted(self._best_results_heap, key=lambda e: -e.objective_value)
        ]

    def all_objective_values(self) -> pd.DataFrame:
        df = pd.DataFrame(
            ((*l, v) for (l, v) in self._all_objective_values),
            columns=self._label_names + ["objective_value"],
        )
        df["objective_value_rank"] = data_ranks(df["objective_value"])
        return df


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


def _search_random_clear_front(
    design: voxart.Design, masks: Masks, rng: np.random.Generator
):
    design.voxels[masks.front_faces(design.goal_locations)] = voxart.EMPTY
    _random_search(design, masks.faces, rng)
    _random_search(design, masks.interior, rng)


def search_filled(
    goal: voxart.Goal,
    strategy: str,
    num_iterations: int,
    top_n: Optional[int],
    obj_func: Optional[ObjectiveFunction] = None,
    masks: Optional[Masks] = None,
    rng: Optional[np.random.Generator] = None,
) -> SearchResults:
    if strategy == "random":
        search_fn = _search_random
    elif strategy == "random_face_first":
        search_fn = _search_random_face_first
    elif strategy == "random_clear_front":
        search_fn = _search_random_clear_front
    else:
        raise ValueError(f"Strategy not known {strategy}")

    if rng is None:
        rng = np.random.default_rng()

    if obj_func is None:
        obj_func = ObjectiveFunction()
    if masks is None:
        masks = Masks(goal.size)
    obj_func.set_masks(masks)

    results = SearchResults(top_n, obj_func, ["form_idx", "is_starting", "iteration"])

    goal_forms = list(goal.alternate_forms())
    pbar = tqdm(goal_forms)
    for form_idx, (goal_form, flips) in enumerate(goal_forms):
        pbar.update(1)
        pbar.set_description(f"Goal form {form_idx} with flips {flips}")
        starting_design = goal_form.create_base_design()
        for axis, flip_val in enumerate(flips):
            if flip_val:
                starting_design.set_goal_location(axis, -1)
        results.add((form_idx, True, -1), starting_design)
        for iter_idx in range(num_iterations):
            design = copy.deepcopy(starting_design)
            search_fn(design, masks, rng)
            results.add((form_idx, False, iter_idx), design)
    pbar.close()

    return results


def get_neighbors(vox: np.typing.ArrayLike, size: int) -> Iterator[np.typing.NDArray]:
    """Generates the neighbors of vox.

    Note that the same array is always returned by the generator, so
    if you need it to be valid after you grab the next, you have to copy it
    """
    # Note that this is intentionally not "asarray" because we need a copy of vox
    # to manipulate for yielding
    vox = np.array(vox)
    if vox.shape != (3,):
        raise ValueError(f"Only suport 3D neightbors, got shape {vox.shape}")
    for axis, delta in itertools.product([0, 1, 2], [-1, 1]):
        newval = vox[axis] + delta
        if newval < 0 or newval >= size:
            continue
        # neighbor = np.copy(vox)
        vox[axis] = newval
        yield vox
        vox[axis] -= delta


@dataclass(order=True)
class _PathEntry:
    vox: Tuple[int, int, int] = field(compare=False)
    parent: Optional[Tuple[int, int, int]] = field(compare=False)
    distance: int


class NoPathError(Exception):
    pass


def get_shortest_path_to_targets(
    design: voxart.Design,
    masks: Masks,
    starting: List[Tuple[int, int, int]],
    targets: Set[Tuple[int, int, int]],
    allowed_mask: Optional[np.typing.NDArray],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Tuple[int, int, int], int, List[Tuple[int, int, int]]]:
    """Given a set of targets, find the shortest path to the edge of the design.

    If there are multiple targets with shortest paths or multiple shortest paths,
    a random one will be returned.
    I don't know if it is uniformly random or not. Probably not.
    We use the rng to randomly pick an edge voxel to start from.
    """
    if rng is None:
        rng = np.random.default_rng()

    starting_idx = rng.integers(len(starting))
    starting_point = starting[starting_idx]

    frontier = [_PathEntry(starting_point, None, 0)]
    visited = {}
    while True:
        try:
            entry = heapq.heappop(frontier)
        except IndexError:
            raise NoPathError

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
            if allowed_mask is not None and not allowed_mask[neighbor]:
                continue
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
    top_n: Optional[int],
    allow_obstructing: bool = True,
    obj_func: Optional[ObjectiveFunction] = None,
    masks: Optional[Masks] = None,
    rng: Optional[np.random.Generator] = None,
) -> SearchResults:
    if rng is None:
        rng = np.random.default_rng()

    if obj_func is None:
        obj_func = ObjectiveFunction()
    if masks is None:
        masks = Masks(starting_design)
    obj_func.set_masks(masks)

    allowed_mask = None
    if allow_obstructing:
        allowed_mask = None
    else:
        allowed_mask = ~starting_design.obstructing_voxels()

    edge_indices = list(zip(*np.where(masks.edges)))
    results = SearchResults(top_n, obj_func, ["iteration", "num_connectors", "failure"])
    for iter_idx in range(num_iterations):
        design = copy.deepcopy(starting_design)

        pending_vox = set(
            (x, y, z)
            for x, y, z in zip(
                *np.where((design.voxels == voxart.FILLED) & ~masks.edges)
            )
        )
        try:
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
                    design,
                    masks,
                    starting=edge_indices,
                    targets=active_subset,
                    allowed_mask=allowed_mask,
                    rng=rng,
                )
                assert target in pending_vox
                add_path_as_connectors(design, path)
                pending_vox.remove(target)

            results.add((iter_idx, design.num_connectors(), False), design)
        except NoPathError:
            design.failure = True
            results.add((iter_idx, design.num_connectors(), True), design)

    return results


def connect_edges(
    design: voxart.Design, masks: Masks, rng: Optional[np.random.Generator] = None
):
    """Connect every edge to an interior block (filled or connector).

    This is probably not what you want to do, but leaving it here in case.
    See connect_faces instead.
    """
    if rng is None:
        rng = np.random.default_rng()

    targets = (design.voxels == voxart.FILLED) | (design.voxels == voxart.CONNECTOR)
    targets &= masks.interior
    targets = set(zip(*np.where(targets)))
    allowed_mask = ~masks.edges

    for edge in masks.single_edges():
        target, distance, path = get_shortest_path_to_targets(
            design,
            masks,
            allowed_mask=allowed_mask | edge,
            starting=list(zip(*np.where(edge))),
            targets=targets,
            rng=rng,
        )
        if distance == 0:
            continue
        add_path_as_connectors(design, path)


def connect_faces(
    design: voxart.Design,
    masks: Optional[Masks] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Connect every face to an interior block (filled or connector)."""
    if rng is None:
        rng = np.random.default_rng()

    if masks is None:
        masks = Masks(design)

    targets = (design.voxels == voxart.FILLED) | (design.voxels == voxart.CONNECTOR)
    targets &= masks.interior
    targets = set(zip(*np.where(targets)))

    for face in masks.full_faces():
        target, distance, path = get_shortest_path_to_targets(
            design,
            masks,
            allowed_mask=face | masks.interior,
            starting=list(zip(*np.where(face & masks.edges))),
            targets=targets,
            rng=rng,
        )
        if distance == 0:
            continue
        add_path_as_connectors(design, path)


def is_vox_unsupported(design: voxart.Design, vox: Tuple[int, int, int]):
    if vox == tuple(design.bottom_location * (design.size - 1)):
        return False

    # This may seem like a pretty dumb way to write this, but the more compact
    # way that I wrote it was 4 times slower than this and it turned out to be a
    # significant fraction of the total run time. This was the fastest I came up with.

    diff = design.bottom_location[0] * 2 - 1
    neigh_vox = (vox[0] + diff, vox[1], vox[2])
    if neigh_vox[0] != -1 and neigh_vox[0] != design.size:
        if design.voxels[neigh_vox] != voxart.EMPTY:
            return False

    diff = design.bottom_location[1] * 2 - 1
    neigh_vox = (vox[0], vox[1] + diff, vox[2])
    if neigh_vox[1] != -1 and neigh_vox[1] != design.size:
        if design.voxels[neigh_vox] != voxart.EMPTY:
            return False

    diff = design.bottom_location[2] * 2 - 1
    neigh_vox = (vox[0], vox[1], vox[2] + diff)
    if neigh_vox[2] != -1 and neigh_vox[2] != design.size:
        if design.voxels[neigh_vox] != voxart.EMPTY:
            return False

    return True


def count_unsupported(design: voxart.Design):
    """Counts the number of filled or connector blocks that are unsupported.

    Unsupported means that none of the 3 neighbors in the direction of the
    design.bottom_location are filled or connector blocks.
    """
    if design.bottom_location is None:
        raise ValueError("Require bottom_location to be set")

    unsupported = 0
    for vox in itertools.product(range(design.size), repeat=3):
        if design.voxels[vox] == voxart.EMPTY:
            continue
        if is_vox_unsupported(design, vox):
            unsupported += 1

    return unsupported


def search_bottom_location(
    design: voxart.Design,
    obj_func: Optional[ObjectiveFunction] = None,
    masks: Optional[Masks] = None,
) -> SearchResults:
    if obj_func is None:
        obj_func = ObjectiveFunction()
    if masks is None:
        masks = Masks(design)
    obj_func.set_masks(masks)

    # Keep all 8 bottom_locations
    results = SearchResults(8, obj_func, ["bottom_location"])
    for bottom_location in itertools.product([0, 1], repeat=3):
        design.bottom_location = bottom_location

        results.add((str(bottom_location),), copy.deepcopy(design))

    return results


@dataclass
class SearchOptions:
    filled_batch_size: int = 3
    filled_num_batches: int = 5
    filled_strategy: str = "random_clear_front"
    filled_num_to_pursue: int = 20
    connector_num_iterations_per: int = 30
    connector_allow_obstructing: bool = False
    connector_frac: float = 0.4
    top_n: int = 20
    obj_func: Optional[ObjectiveFunction] = None


class UniqueDesignIDer:
    def __init__(self):
        self._seen_map: Dict[int, int] = {}

    def add(self, design: voxart.Design) -> Tuple[bool, int]:
        hash_val = hash(design)
        if hash_val in self._seen_map:
            return False, self._seen_map[hash_val]
        uid = len(self._seen_map)
        self._seen_map[hash_val] = uid
        return True, uid


def search(
    goal: voxart.Goal, opts: SearchOptions, rng: Optional[np.random.Generator] = None
) -> Tuple[Optional[SearchResults], Optional[SearchResults]]:
    obj_func = opts.obj_func
    if obj_func is None:
        obj_func = ObjectiveFunction()
    masks = Masks(goal.size)
    obj_func.set_masks(masks)

    if rng is None:
        rng = np.random.default_rng()

    print(opts)

    complete_results = None
    all_filled_results = None
    seen_design_filled = UniqueDesignIDer()
    seen_design_complete = UniqueDesignIDer()

    pbar = tqdm(total=opts.filled_num_batches * opts.filled_num_to_pursue)
    for batch_idx in range(opts.filled_num_batches):
        pbar.set_description(f"Batch {batch_idx}/{opts.filled_num_batches}")
        filled_results = search_filled(
            goal,
            strategy=opts.filled_strategy,
            num_iterations=opts.filled_batch_size,
            top_n=opts.filled_num_to_pursue,
            obj_func=obj_func,
            masks=masks,
            rng=rng,
        )
        for idx_in_batch, (filled_labels, filled_design) in enumerate(
            filled_results.best()
        ):
            pbar.update(1)
            filled_is_unique, filled_uid = seen_design_filled.add(filled_design)
            # We lazily initialize all_filled_results to get the labels right
            if all_filled_results is None:
                all_filled_results = SearchResults(
                    0,  # Don't keep anything, we just want the stats
                    obj_func,
                    [
                        "batch_idx",
                        "idx_in_batch",
                        *["filled_" + x for x in filled_results.label_names],
                        "filled_is_unique",
                        "filled_uid",
                    ],
                )
            all_filled_results.add(
                (
                    batch_idx,
                    idx_in_batch,
                    *filled_labels,
                    filled_is_unique,
                    filled_uid,
                ),
                filled_design,
            )

            if not filled_is_unique:
                continue

            connector_results = search_connectors(
                filled_design,
                num_iterations=opts.connector_num_iterations_per,
                top_n=None,
                allow_obstructing=opts.connector_allow_obstructing,
                obj_func=obj_func,
                masks=masks,
                rng=rng,
            )
            conn_to_pursue = connector_results.best()
            conn_to_pursue = conn_to_pursue[
                : round(len(conn_to_pursue) * opts.connector_frac)
            ]
            for idx_in_connector, (connector_labels, connector_design) in enumerate(
                conn_to_pursue
            ):
                bottom_location_results = search_bottom_location(
                    connector_design, obj_func=obj_func, masks=masks
                )

                # We lazily initialize complete_results so that we can get all the right labels
                # from the sub complete_results
                if complete_results is None:
                    complete_results = SearchResults(
                        opts.top_n,
                        obj_func,
                        [
                            "batch_idx",
                            "idx_in_batch",
                            *["filled_" + x for x in filled_results.label_names],
                            "filled_is_unique",
                            "filled_uid",
                            "idx_in_connector",
                            *["conn_" + x for x in connector_results.label_names],
                            "idx_in_bottom_location",
                            *bottom_location_results.label_names,
                            "is_unique",
                            "uid",
                        ],
                    )
                for idx_in_bottom_location, (
                    bottom_location_labels,
                    design,
                ) in enumerate(bottom_location_results.best()):

                    complete_is_unique, complete_uid = seen_design_complete.add(design)

                    complete_results.add(
                        (
                            batch_idx,
                            idx_in_batch,
                            *filled_labels,
                            filled_is_unique,
                            filled_uid,
                            idx_in_connector,
                            *connector_labels,
                            idx_in_bottom_location,
                            *bottom_location_labels,
                            complete_is_unique,
                            complete_uid,
                        ),
                        design,
                    )

    pbar.close()

    return all_filled_results, complete_results
