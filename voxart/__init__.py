# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

import copy
import itertools
from typing import Iterator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .geometry import *
from .search import *

EMPTY: int = 0
CONNECTOR: int = 1
FILLED: int = 2


class Design:
    def __init__(self, voxels: np.typing.ArrayLike):
        voxels = np.asarray(voxels)
        if len(voxels.shape) != 3:
            raise ValueError(
                f"Voxels for design must have 3D shape, got {voxels.shape}"
            )
        if not (
            voxels.shape[0] == voxels.shape[1] and voxels.shape[1] == voxels.shape[2]
        ):
            raise ValueError(
                f"Voxels for design must have equal dims, got {voxels.shape}"
            )

        self._voxels = np.copy(voxels.astype(int))

    @staticmethod
    def from_size(size) -> Design:
        return Design(np.zeros((size, size, size)))

    @staticmethod
    def from_npy(fn) -> Design:
        return Design(np.load(fn))

    def save_npy(self, fn: str):
        np.save(fn, self._voxels, allow_pickle=False)

    def __eq__(self, other):
        if self._voxels.shape != other._voxels.shape:
            return False
        return np.all(self._voxels == other._voxels)

    def __hash__(self):
        return hash(self._voxels.tobytes())

    @property
    def size(self) -> int:
        return self._voxels.shape[0]

    @property
    def voxels(self) -> np.typing.NDArray:
        return self._voxels

    def projection(self, axis: int) -> np.typing.NDArray:
        return self._voxels.max(axis)

    def projections(self) -> Iterator[np.typing.NDArray]:
        for axis in range(3):
            yield self.projection(axis)

    def slice(self, axis: int, idx: int) -> np.typing.NDArray:
        return np.take(self._voxels, idx, axis=axis)

    def slices(self, axis: int) -> Iterator[np.typing.NDArray]:
        for i in range(self.size):
            yield self.slice(axis, i)

    def num_filled(self) -> int:
        return np.sum(self._voxels == FILLED)

    def num_connectors(self) -> int:
        return np.sum(self._voxels == CONNECTOR)

    def add_frame(self, offset: int = 0):
        if offset == 0:
            line_slice = slice(None)
        else:
            line_slice = slice(offset, -offset)
        for axis in range(3):
            for axis_idx in [0, -1]:
                for indexer in [
                    [(offset, -1 - offset), line_slice],
                    [line_slice, (offset, -1 - offset)],
                ]:
                    indexer.insert(axis, axis_idx)
                    self.voxels[tuple(indexer)] = voxart.FILLED

    def find_removable_slow(self) -> np.typing.NDArray:
        """Finds all voxels that can be removed without changing projections.

        I think this will be a slow implementation because it repeats lots of work
        of summing along axis that can probably be done once. But I need to test this.
        """
        out = np.full((self.size, self.size, self.size), False)
        for x, y, z in itertools.product(
            range(self.size), range(self.size), range(self.size)
        ):
            if not self._voxels[x, y, z]:
                continue
            if (
                np.sum(self._voxels[x, y, :] == FILLED) > 1
                and np.sum(self._voxels[:, y, z] == FILLED) > 1
                and np.sum(self._voxels[x, :, z] == FILLED) > 1
            ):
                out[x, y, z] = True
        return out

    def find_removable(self) -> np.typing.NDArray:
        """Finds all voxels that can be removed without changing projections."""
        sums = [
            np.expand_dims(np.sum(self._voxels == FILLED, axis=axis), axis=axis)
            for axis in range(3)
        ]
        min_array = np.minimum(sums[0], np.minimum(sums[1], sums[2]))
        # You have to and this with the original array because the sums across all axes can be
        # be larger than 1 even if that voxel itself is not set.
        return np.logical_and(min_array > 1, self._voxels == FILLED)

    def projections_fig(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        for axis, ax in enumerate(axes):
            ax.imshow(
                self.projection(axis),
                cmap="Greys",
                interpolation="none",
                vmin=0,
                vmax=2,
            )
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            ax.set_xticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
            ax.set_yticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
            ax.grid(visible=True)
        plt.close()
        return fig

    def slices_fig(self) -> plt.Figure:
        fig, axes = plt.subplots(3, self.size, figsize=(2 * self.size, 6))
        for axis in range(3):
            for i, slc in enumerate(self.slices(axis)):
                ax = axes[axis, i]
                ax.imshow(slc, cmap="Greys", interpolation="none", vmin=0, vmax=2)
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False,
                )
                ax.set_xticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
                ax.set_yticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
                ax.grid(visible=True)
        plt.close()
        return fig


class Goal:
    def __init__(self, arr: np.typing.ArrayLike):
        arr = np.asarray(arr)
        if arr.shape[0] != 3:
            raise ValueError(
                f"Goals expect first dimension to have size 3, got {arr.shape}"
            )
        if arr.shape[1] != arr.shape[2]:
            raise ValueError(f"Goals expect square dimensions, got {arr.shape}")
        if (np.sum(arr == voxart.EMPTY) + np.sum(arr == voxart.FILLED)) != arr.size:
            raise ValueError(
                "Goals must contain only EMPTY and FILLED, got:"
                + str(arr[(arr != EMPTY) & (arr != FILLED)])
            )
        self._goals = np.copy(arr)

    @staticmethod
    def from_size(size: int) -> Goal:
        return Goal(np.full((3, size, size), EMPTY))

    @staticmethod
    def from_arrays(
        arr0: np.typing.ArrayLike, arr1: np.typing.ArrayLike, arr2: np.typing.ArrayLike
    ) -> Goal:
        arr0 = np.asarray(arr0)
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        if arr0.shape != arr1.shape or arr1.shape != arr2.shape:
            raise ValueError(
                f"Goals needs arrays of the same shape, "
                f"got {arr0.shape} {arr1.shape} {arr2.shape}"
            )
        return Goal(np.stack([arr0, arr1, arr2]))

    @staticmethod
    def from_image(img: Image.Image):
        arr = np.array(img.convert(mode="L"))
        arr[arr < 128] = FILLED
        arr[arr >= 128] = EMPTY

        sz = arr.shape[1]
        return voxart.Goal(arr.reshape((3, sz, sz)))

    def __eq__(self, other):
        if self._goals.shape != other._goals.shape:
            return False
        return np.all(self._goals == other._goals)

    def __hash__(self):
        return hash(self._goals.tobytes())

    @property
    def size(self) -> int:
        return self._goals.shape[1]

    def goal(self, goal_idx: int) -> np.typing.NDArray:
        return self._goals[goal_idx, :, :]

    def to_image(self):
        sz = self.size
        out = np.reshape(self._goals, (3 * sz, sz)).astype(np.uint8)
        out[out == EMPTY] = 255
        out[out == FILLED] = 0
        return Image.fromarray(out, mode="L")

    def add_frame(self):
        self._goals[:, (0, -1), :] = FILLED
        self._goals[:, :, (0, -1)] = FILLED

    def alternate_forms(self, include_flips: bool = True) -> Iterator[Goal]:
        """Produce alternate forms that produce equivalent projectiosn.

        include_flips is just an argument for testing, should not be used in practive
        """
        seen = set()
        if include_flips:
            flips = [False, True]
        else:
            flips = [False]
        for arr1_flip, arr1_rot, arr2_flip, arr2_rot in itertools.product(
            flips, range(4), flips, range(4)
        ):
            arr1 = self._goals[1, :, :]
            arr2 = self._goals[2, :, :]
            if arr1_flip:
                arr1 = np.flip(arr1, axis=0)
            if arr2_flip:
                arr2 = np.flip(arr2, axis=0)
            arr1 = np.rot90(arr1, k=arr1_rot)
            arr2 = np.rot90(arr2, k=arr2_rot)
            goal = Goal.from_arrays(self._goals[0, :, :], arr1, arr2)
            old_len = len(seen)
            seen.add(goal)
            if len(seen) > old_len:
                yield goal

    def create_base_design(self) -> Design:
        design = Design.from_size(self._goals.shape[1])
        design.voxels[:, :, :] = FILLED
        for goal_idx in range(3):
            indices = list(np.where(self._goals[goal_idx] == 0))
            # This seems wrong -- I have to know the end idx to make this slice?
            indices.insert(goal_idx, slice(self._goals.shape[1]))
            design.voxels[tuple(indices)] = EMPTY

        return design

    def fig(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        for goal_idx, ax in enumerate(axes):
            ax.imshow(
                self._goals[goal_idx, :, :],
                cmap="binary",
                interpolation="none",
                vmin=0,
                vmax=2,
            )
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            ax.set_xticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
            ax.set_yticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
            ax.grid(visible=True)
        plt.close()
        return fig
