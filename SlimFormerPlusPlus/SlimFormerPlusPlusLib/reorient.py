# SPDX-License-Identifier: Apache-2.0
"""Bring a volume into the voxel-axis orientation the model was trained on.

The AMOS training images all have the ITK (LPS) direction diag(1, -1, 1): voxel
index i runs towards the patient's left, j towards anterior and k towards
superior. nnU-Net ignores the direction matrix, so an image stored with other
axis directions (for example DICOM, where j usually runs towards posterior)
would reach the network mirrored or with permuted axes.

``to_model_axes`` permutes and flips the voxel array so that its axes match the
training orientation as closely as possible; it never interpolates, so
``from_model_axes`` restores the original voxel grid exactly.

Arrays use the numpy order of 3D Slicer and SimpleITK, ``array[k, j, i]``.
"""
from __future__ import annotations

import itertools

import numpy as np

# RAS directions of the model's i, j, k axes (left, anterior, superior)
MODEL_AXES_RAS = np.array([[-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])


def _directions_and_spacing(ijk_to_ras):
    m = np.asarray(ijk_to_ras, dtype=float)[:3, :3]
    spacing = np.linalg.norm(m, axis=0)
    if np.any(spacing <= 0):
        raise ValueError('IJK-to-RAS matrix has a zero-length axis')
    return m / spacing, spacing


def axis_mapping(ijk_to_ras):
    """For each model axis m (i, j, k): the source axis perm[m] and whether it is flipped."""
    directions, _ = _directions_and_spacing(ijk_to_ras)      # column a = RAS direction of source axis a
    cos = MODEL_AXES_RAS @ directions                           # cos[m, a]
    perm = max(itertools.permutations(range(3)), key=lambda p: sum(abs(cos[m, p[m]]) for m in range(3)))
    flip = tuple(bool(cos[m, perm[m]] < 0) for m in range(3))
    return tuple(perm), flip


def to_model_axes(array_kji, ijk_to_ras):
    """Return (array in model axes [k, j, i], spacing [k, j, i], mapping)."""
    if array_kji.ndim != 3:
        raise ValueError(f'expected a 3D array, got shape {array_kji.shape}')
    perm, flip = axis_mapping(ijk_to_ras)
    _, spacing = _directions_and_spacing(ijk_to_ras)
    a = np.transpose(array_kji, (2, 1, 0))                      # [i, j, k]
    a = np.transpose(a, perm)
    for m in range(3):
        if flip[m]:
            a = np.flip(a, axis=m)
    model_spacing_ijk = spacing[list(perm)]
    return (np.ascontiguousarray(np.transpose(a, (2, 1, 0))),
            [float(s) for s in model_spacing_ijk[::-1]],
            {'perm': perm, 'flip': flip})


def from_model_axes(array_model_kji, mapping):
    """Inverse of ``to_model_axes`` for an array on the model grid."""
    perm, flip = mapping['perm'], mapping['flip']
    a = np.transpose(array_model_kji, (2, 1, 0))                # model [i, j, k]
    for m in range(3):
        if flip[m]:
            a = np.flip(a, axis=m)
    a = np.transpose(a, np.argsort(perm))                        # source [i, j, k]
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))
