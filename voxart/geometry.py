# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

from typing import Tuple

import numpy as np
import stl

import voxart

_VOX_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

_VOX_TRI_IDX = [
    # bottom
    [0, 3, 1],
    [2, 1, 3],
    # top
    [4, 5, 7],
    [6, 7, 5],
    # front
    [0, 1, 4],
    [5, 4, 1],
    # back
    [3, 7, 2],
    [6, 2, 7],
    # left
    [0, 4, 3],
    [7, 3, 4],
    # right
    [1, 2, 5],
    [6, 5, 2],
]


def vox_to_triangles(vox: Tuple[int, int, int]):
    vox = np.asarray(vox)
    for tri in _VOX_TRI_IDX:
        yield vox + _VOX_VERTICES[tri]


def vox_to_stl(vox: Tuple[int, int, int]):
    out = stl.mesh.Mesh(np.zeros(12, dtype=stl.mesh.Mesh.dtype))
    for i, tri in enumerate(vox_to_triangles(vox)):
        for j in range(3):
            out.vectors[i][j] = tri[j]
    return out


def design_to_stl(design: voxart.Design):
    meshes = []
    for vox in zip(*np.where(design.voxels == voxart.FILLED)):
        meshes.append(vox_to_stl(vox))
    return stl.mesh.Mesh(np.concatenate([m.data for m in meshes]))
