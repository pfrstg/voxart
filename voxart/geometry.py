# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import stl

import voxart

_VOX_VERTICES = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
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

_VOX_FACES = [
    [0, 3, 2, 1],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
]

_RECT_FACES = [
    [0, 1, 2, 3],
    [0, 3, 5, 4],
    [3, 2, 6, 5],
    [2, 1, 7, 6],
    [1, 0, 4, 7],
    [4, 5, 6, 7],
]


def face_to_triangles(vertices: np.typing.ArrayLike) -> Iterable[np.typing.NDArray]:
    """Converts a rectangular face to triangles.

    Vertices must
    * be shape [4, 3]
    * follow the perimeter
    * follwo the right hand rule
    """
    if vertices.shape != (4, 3):
        raise ValueError(f"Unexpected shape {vertices.shape}")
    yield np.array([vertices[1], vertices[2], vertices[0]])
    yield np.array([vertices[3], vertices[0], vertices[2]])


def rect_to_triangles(vertices: np.typing.Arraylike) -> Iterable[np.typing.NDArray]:
    """Convert a rectangular prism to triangles.

    The 8 points must define two opposite faces (first 4 points, second 4 points)
    each in right hand rule order such that the normal vectors face away from each other.
    In addition, the following points must be connected
    0 -> 4
    3 -> 5
    2 -> 6
    1 -> 7
    """
    if vertices.shape != (8, 3):
        raise ValueError(f"Unexpected shape {vertices.shape}")
    for face in _RECT_FACES:
        yield from face_to_triangles(vertices[face])


def vox_to_triangles(vox: Tuple[int, int, int]) -> Iterable[np.typing.NDArray]:
    vox = np.asarray(vox)
    yield from rect_to_triangles(vox + _VOX_VERTICES)


def vox_to_stl(vox: Tuple[int, int, int]) -> stl.mesh.Mesh:
    out = stl.mesh.Mesh(np.zeros(12, dtype=stl.mesh.Mesh.dtype))
    for i, tri in enumerate(vox_to_triangles(vox)):
        for j in range(3):
            out.vectors[i][j] = tri[j]
    return out


def design_to_stl(design: voxart.Design, vox_type: int) -> stl.mesh.Mesh:
    meshes = []
    for vox in zip(*np.where(design.voxels == vox_type)):
        meshes.append(vox_to_stl(vox))
    return stl.mesh.Mesh(np.concatenate([m.data for m in meshes]))


def save_stl_pair(design: voxart.Design, file_stem: str):
    if np.sum(design.voxels == voxart.FILLED):
        design_to_stl(design, voxart.FILLED).save(file_stem + "_filled.stl")
    if np.sum(design.voxels == voxart.CONNECTOR):
        design_to_stl(design, voxart.CONNECTOR).save(file_stem + "_connector.stl")
