# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .search import *


class Design:
    def __init__(self, vox: np.typing.ArrayLike):
        vox = np.asarray(vox)
        if len(vox.shape) != 3:
            raise ValueError(f"Voxels for design must have 3D shape, got {vox.shape}")
        if not (vox.shape[0] == vox.shape[1] and vox.shape[1] == vox.shape[2]):
            raise ValueError(f"Voxels for design must have equal dims, got {vox.shape}")

        self._vox = np.copy(vox.astype(int))

    @staticmethod
    def from_size(size) -> Design:
        return Design(np.zeros((size, size, size)))

    def __eq__(self, other):
        if self._vox.shape != other._vox.shape:
            return False
        return np.all(self._vox == other._vox)

    def __hash__(self):
        return hash(self._vox.tobytes())

    @property
    def size(self) -> int:
        return self._vox.shape[0]

    @property
    def vox(self) -> np.typing.NDArray:
        return self._vox

    def projection(self, axis: int) -> np.typing.NDArray:
        return self._vox.max(axis)

    def projections(self) -> Iterator[np.typing.NDArray]:
        for axis in range(3):
            yield self.projection(axis)

    def slice(self, axis: int, idx: int) -> np.typing.NDArray:
        return np.take(self._vox, idx, axis=axis)

    def slices(self, axis: int) -> Iterator[np.typing.NDArray]:
        for i in range(self.size):
            yield self.slice(axis, i)

    def find_removable_slow(self) -> np.typing.NDArray:
        """Finds all voxels that can be removed without changing projections.

        I think this will be a slow implementation because it repeats lots of work
        of summing along axis that can probably be done once. But I need to test this.
        """
        out = np.full((self.size, self.size, self.size), False)
        for x, y, z in itertools.product(range(self.size), range(self.size), range(self.size)):
            if not self._vox[x, y, z]:
                continue
            if (np.sum(self._vox[x, y, :]) > 1 and
                np.sum(self._vox[:, y, z]) > 1 and
                np.sum(self._vox[x, :, z]) > 1):
                out[x, y, z] = True
        return out

    def projections_fig(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        for axis, ax in enumerate(axes):
            #ax.imshow(vox[i, :, :], cmap="binary", interpolation="none")
            ax.imshow(self.projection(axis), cmap="binary", interpolation="none", vmin=0, vmax=1)
            ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
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
                #ax.imshow(vox[i, :, :], cmap="binary", interpolation="none")
                ax.imshow(slc, cmap="binary", interpolation="none", vmin=0, vmax=1)
                ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
                ax.set_xticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
                ax.set_yticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
                ax.grid(visible=True)
        plt.close()
        return fig


class Goal:
    def __init__(self, arr: np.typing.ArrayLike):
        arr = np.asarray(arr)
        if arr.shape[0] != 3:
            raise ValueError(f"Goals expect first dimension to have size 3, got {arr.shape}")
        if arr.shape[1] != arr.shape[2]:
            raise ValueError(f"Goals expect square dimensions, got {arr.shape}")
        self._goals = np.copy(arr)

    @staticmethod
    def from_size(size: int) -> Goal:
        return Goal(np.zeros((3, size, size)))

    @staticmethod
    def from_arrays(arr0: np.typing.ArrayLike,
                    arr1: np.typing.ArrayLike,
                    arr2: np.typing.ArrayLike) -> Goal:
        arr0 = np.asarray(arr0)
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        if arr0.shape != arr1.shape or arr1.shape != arr2.shape:
            raise ValueError(f"Goals needs arrays of the same shape, "
                             f"got {arr0.shape} {arr1.shape} {arr2.shape}")
        return Goal(np.stack([arr0, arr1, arr2]))

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
        return self._goals[goal_idx, : , :]

    def add_frame(self):
        self._goals[:, (0, -1), :] = 1
        self._goals[:, :, (0, -1)] = 1

    def alternate_forms(self, include_flips: bool=True) -> Iterator[Goal]:
        """Produce alternate forms that produce equivalent projectiosn.

        include_flips is just an argument for testing, should not be used in practive
        """
        seen = set()
        if include_flips:
            flips = [False, True]
        else:
            flips = [False]
        for arr1_flip, arr1_rot, arr2_flip, arr2_rot in itertools.product(
                flips,  range(4), flips, range(4)):
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
        design.vox[:, :, :] = 1
        for goal_idx in range(3):
            indices = list(np.where(self._goals[goal_idx] == 0))
            # This seems wrong -- I have to know the end idx to make this slice?
            indices.insert(goal_idx, slice(self._goals.shape[1]))
            design.vox[tuple(indices)] = 0

        return design

    def fig(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        for goal_idx, ax in enumerate(axes):
            #ax.imshow(vox[i, :, :], cmap="binary", interpolation="none")
            ax.imshow(self._goals[goal_idx, :, :], cmap="binary", interpolation="none",
                      vmin=0, vmax=1)
            ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            ax.set_xticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
            ax.set_yticks(np.linspace(-0.5, self.size - 0.5, self.size + 1))
            ax.grid(visible=True)
        plt.close()
        return fig
