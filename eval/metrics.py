"""Functions for metrics related to 2D and 3D bounding boxes."""

# pylint: disable=invalid-name,missing-docstring,assignment-from-no-return,logging-format-interpolation

import logging

import numpy as np

from bbox3d import BBox3D
from geometry import polygon_area, polygon_collision, polygon_intersection

logger = logging.getLogger(__name__)



def iou_3d(a: BBox3D, b: BBox3D):
    """
    Compute the Intersection over Union (IoU) of a pair of 3D bounding boxes.

    Alias for `jaccard_index_3d`.
    """
    return jaccard_index_3d(a, b)


def jaccard_index_3d(a: BBox3D, b: BBox3D):
    """
    Compute the Jaccard Index / Intersection over Union (IoU) of a pair of 3D bounding boxes.
    We compute the IoU using the top-down bird's eye view of the boxes.

    **Note**: We follow the KITTI format and assume only yaw rotations (along z-axis).

    Args:
        a (:py:class:`BBox3D`): 3D bounding box.
        b (:py:class:`BBox3D`): 3D bounding box.

    Returns:
        :py:class:`float`: The IoU of the 2 bounding boxes.
    """
    # check if the two boxes don't overlap
    if not polygon_collision(a.p[0:4, 0:2], b.p[0:4, 0:2]):
        return np.round_(0, decimals=5)

    intersection_points = polygon_intersection(a.p[0:4, 0:2], b.p[0:4, 0:2])
    inter_area = polygon_area(intersection_points)

    zmax = np.minimum(a.cz, b.cz)
    zmin = np.maximum(a.cz - a.h, b.cz - b.h)

    inter_vol = inter_area * np.maximum(0, zmax-zmin)

    a_vol = a.l * a.w * a.h
    b_vol = b.l * b.w * b.h

    union_vol = (a_vol + b_vol - inter_vol)

    iou = inter_vol / union_vol

    # set nan and +/- inf to 0
    if np.isinf(iou) or np.isnan(iou):
        iou = 0

    return np.round_(iou, decimals=5)
