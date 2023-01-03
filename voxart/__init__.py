import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Design:
    def __init__(self, vox):
        vox = np.asarray(vox)
        if len(vox.shape) != 3:
            raise ValueError(f"Voxels for design must have 3D shape, got {vox.shape}")
        if not (vox.shape[0] == vox.shape[1] and vox.shape[1] == vox.shape[2]):
            raise ValueError(f"Voxels for design must have equal dims, got {vox.shape}")

        self._vox = np.copy(vox)

    @staticmethod
    def from_size(size):
        return Design(np.zeros((size, size, size)))

    @property
    def size(self):
        return self._vox.shape[0]

    @property
    def vox(self):
        return self._vox

    def projection(self, axis):
        return self._vox.max(axis)

    def projections(self):
        for axis in range(3):
            yield self.projection(axis)

    def slices(self, axis):
        for i in range(self.size):
            yield np.take(self._vox, i, axis=axis)

    def projections_fig(self):
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

    def slices_fig(self):
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
    def __init__(self, arr):
        arr = np.asarray(arr)
        if arr.shape[0] != 3:
            raise ValueError(f"Goals expect first dimension to have size 3, got {arr.shape}")
        if arr.shape[1] != arr.shape[2]:
            raise ValueError(f"Goals expect square dimensions, got {arr.shape}")
        self._goals = np.copy(arr)

    @staticmethod
    def from_size(size):
        return Goal(np.zeros((3, size, size)))

    @staticmethod
    def from_arrays(arr0, arr1, arr2):
        arr0 = np.asarray(arr0)
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        if arr0.shape != arr1.shape or arr1.shape != arr2.shape:
            raise ValueError(f"Goals needs arrays of the same shape, "
                             f"got {arr0.shape} {arr1.shape} {arr2.shape}")
        return Goal(np.stack([arr0, arr1, arr2]))

    @property
    def size(self):
        return self._goals.shape[1]

    def goal(self, goal_idx):
        return self._goals[goal_idx, : , :]

    def add_frame(self):
        self._goals[:, (0, -1), :] = 1
        self._goals[:, :, (0, -1)] = 1

    def rotations(self):
        # TODO: maybe I should be smart and not generate equivalent figures
        for arr1_rot, arr2_rot in itertools.product(range(4), range(4)):
            yield Goal.from_arrays(self._goals[0, :, :],
                                   np.rot90(self._goals[1, :, :], k=arr1_rot),
                                   np.rot90(self._goals[2, :, :], k=arr2_rot))

    def create_base_design(self):
        design = Design.from_size(self._goals.shape[1])
        design.vox[:, :, :] = 1
        for goal_idx in range(3):
            indices = list(np.where(self._goals[goal_idx] == 0))
            # This seems wrong -- I have to know the end idx to make this slice?
            indices.insert(goal_idx, slice(self._goals.shape[1]))
            design.vox[tuple(indices)] = 0

        return design

    def fig(self):
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
