# Copyright 2023, Patrick Riley, github: pfrstg

from __future__ import annotations

import functools
import glob
import math
import os
import subprocess
from typing import Iterable, List, Optional, Tuple

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


class _ConnectorStrutInfo:
    """Info for creating the small struts as connectors

    Attributes:
      vox_idx_delta: tuple with the change in vox indexer
      neighbor_vec: Length 1 vector pointing at the neighbor
      face_vecs: List of 4 length 1 vectors from the center of the
                  small face of the strut at the center of the voxel
                  and pointing to the corners.  To create the corner
                  fro the face at the neighboring voxel, you have to
                  iterate in order 0, -1, -2, -3

    """

    def __init__(
        self, vox_idx_delta: np.typing.ArrayLike, face_vecs: List[np.typing.NDArray]
    ):
        self.vox_idx_delta = np.asarray(vox_idx_delta, dtype=int)
        if self.vox_idx_delta.shape != (3,):
            raise ValueError(f"vox_idx_delta must be shape (3,), got {vox_idx_delta}")
        self.neighbor_vec = np.asarray(vox_idx_delta, dtype=float)
        self.neighbor_vec /= np.linalg.norm(self.neighbor_vec)
        if len(face_vecs) != 4:
            raise ValueError(f"Face vecs mist be len 4, got {face_vecs}")
        self.face_vecs = [np.asarray(v, dtype=float) for v in face_vecs]
        for i in range(4):
            self.face_vecs[i] /= np.linalg.norm(self.face_vecs[i])


# I'm sure there is some smart geometry way to generate these automatically
# so that the faces obey the right hand rule. But I couldn't figure it out
# so I just wrote them all out by hand.
_CONNECTOR_STRUT_INFO = [
    _ConnectorStrutInfo(
        (1, 0, 0),
        [
            [0, 1, 1],
            [0, 1, -1],
            [0, -1, -1],
            [0, -1, 1],
        ],
    ),
    _ConnectorStrutInfo(
        (-1, 0, 0),
        [
            [0, 1, 1],
            [0, -1, 1],
            [0, -1, -1],
            [0, 1, -1],
        ],
    ),
    _ConnectorStrutInfo(
        (0, 1, 0),
        [
            [1, 0, 1],
            [-1, 0, 1],
            [-1, 0, -1],
            [1, 0, -1],
        ],
    ),
    _ConnectorStrutInfo(
        (0, -1, 0),
        [
            [1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [-1, 0, 1],
        ],
    ),
    _ConnectorStrutInfo(
        (0, 0, 1),
        [
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [-1, 1, 0],
        ],
    ),
    _ConnectorStrutInfo(
        (0, 0, -1),
        [
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
        ],
    ),
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


def connector_struct(
    from_vox: Tuple[int, int, int], strut_info: _ConnectorStrutInfo, strut_width: float
):
    center = np.array(from_vox) + np.array([0.5, 0.5, 0.5])
    inside_face_center = center - strut_width / 2 * strut_info.neighbor_vec
    outside_face_center = center + 0.5 * strut_info.neighbor_vec
    rect_vertices = []
    face_vec_len = strut_width * math.sqrt(2) / 2
    for i in range(4):
        rect_vertices.append(
            inside_face_center + face_vec_len * strut_info.face_vecs[i]
        )
    for i in range(0, -4, -1):
        rect_vertices.append(
            outside_face_center + face_vec_len * strut_info.face_vecs[i]
        )
    yield from rect_to_triangles(np.array(rect_vertices))


def make_connector_strut_test_stl():
    triangles = list(vox_to_triangles((0, 0, 0)))
    for i, strut_info in enumerate(_CONNECTOR_STRUT_INFO):
        triangles.extend(connector_struct((i + 1, i + 1, i + 1), strut_info, 0.2))
    return triangles_to_stl(triangles)


def vox_to_triangles(vox: Tuple[int, int, int]) -> Iterable[np.typing.NDArray]:
    vox = np.asarray(vox)
    yield from rect_to_triangles(vox + _VOX_VERTICES)


def triangles_to_stl(triangles: List[np.typing.NDArray]) -> stl.mesh.Mesh:
    out = stl.mesh.Mesh(np.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        for j in range(3):
            out.vectors[i][j] = tri[j]
    return out


def vox_to_stl(vox: Tuple[int, int, int]) -> stl.mesh.Mesh:
    return triangles_to_stl(list(vox_to_triangles(vox)))


def design_to_cube_stl(design: voxart.Design, vox_type: int) -> stl.mesh.Mesh:
    meshes = []
    for vox in zip(*np.where(design.voxels == vox_type)):
        meshes.append(vox_to_stl(vox))
    return stl.mesh.Mesh(np.concatenate([m.data for m in meshes]))


def design_to_connector_strut_stl(
    design: voxart.Design, strut_width: float
) -> stl.mesh.Mesh:
    triangles = []
    for vox in zip(*np.where(design.voxels == voxart.CONNECTOR)):
        for info in _CONNECTOR_STRUT_INFO:
            neighbor_vox = vox + info.vox_idx_delta
            if np.any(neighbor_vox < 0) or np.any(neighbor_vox >= design.size):
                continue
            if design.voxels[tuple(neighbor_vox)] == voxart.EMPTY:
                continue
            triangles.extend(connector_struct(vox, info, strut_width))
    return triangles_to_stl(triangles)


@functools.lru_cache(maxsize=None)
def _locate_prusa_slicer() -> Optional[str]:
    mac_location = "/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer"
    colab_glob = "*/bin/prusa-slicer"
    linux_location = "/usr/local/bin/prusa-slicer"

    if os.path.exists(mac_location):
        return mac_location
    if os.path.exists(linux_location):
        return linux_location
    matches = glob.glob(colab_glob)
    if matches:
        return matches[0]
    return None


def transform_stl_to_stand_on_point(
    mesh: stl.mesh.Mesh, bottom_location: np.typing.ArrayLike, size: float
):
    """Mutates the given mesh to so bottom_location point in the cube is at the bottom.

    bottom_location must be a 0,1 shpae (3,) vector (see Design.bottom_location)
    size will generally be Design.size
    """
    bottom_location = np.asarray(bottom_location, dtype=np.int32)
    if not np.all((bottom_location == 1) | (bottom_location == 0)):
        raise ValueError(f"Bad bottom_lcoation {bottom_location}")
    bottom_pt = bottom_location * size
    mesh.vectors -= bottom_pt
    rot_from_pt = ((bottom_location + 1) % 2) * size - bottom_pt
    target = np.array([0, 0, 1])
    cp = np.cross(rot_from_pt, target)
    angle = np.arccos(np.dot(rot_from_pt, target) / np.linalg.norm(rot_from_pt))
    # I really don't know why we have the reverse the cross product here.
    # That seems like some diagreement on right hand rule stuff, but this works.
    mesh.rotate(axis=-cp, theta=angle)


def save_model_files(
    design: voxart.Design,
    file_stem: str,
    connector_style: str = "cube",
    strut_width: float = 0.2,
    scale: float = 1.0,
    write_merged: bool = True,
):
    filled_stl = design_to_cube_stl(design, voxart.FILLED)
    filled_stl.vectors *= scale

    if np.sum(design.voxels == voxart.CONNECTOR) == 0:
        connector_stl = None
    elif connector_style == "cube":
        connector_stl = design_to_cube_stl(design, voxart.CONNECTOR)
    elif connector_style == "strut":
        connector_stl = design_to_connector_strut_stl(design, strut_width)
    else:
        raise ValueError(f"Bad connector_style {connector_style}")
    if connector_stl is not None:
        connector_stl.vectors *= scale

    # Rotate the STL if requested by the Design
    if design.bottom_location is not None:
        transform_stl_to_stand_on_point(
            filled_stl, design.bottom_location, design.size * scale
        )
        if connector_stl is not None:
            transform_stl_to_stand_on_point(
                connector_stl, design.bottom_location, design.size * scale
            )

    if connector_stl is not None:
        joint_stl = stl.mesh.Mesh(np.concatenate([filled_stl.data, connector_stl.data]))
    else:
        joint_stl = filled_stl

    filled_stl_fn = file_stem + "_filled.stl"
    connector_stl_fn = file_stem + "_connector.stl"

    joint_stl.save(file_stem + ".stl")
    filled_stl.save(filled_stl_fn)
    if connector_stl:
        connector_stl.save(connector_stl_fn)

    if write_merged:
        slicer_path = _locate_prusa_slicer()
        if not slicer_path:
            raise ValueError("Could not find Prusa Slicer")

        for file_format in ["3mf", "obj"]:
            args = [
                slicer_path,
                "--export-" + file_format,
                "--merge",
                "--dont-arrange",
                "--output",
                file_stem + "." + file_format,
                filled_stl_fn,
            ]
            if connector_stl is not None:
                args.append(connector_stl_fn)
            subprocess.run(args, check=True)
