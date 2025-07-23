#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-07-08
Description: Find unique symmetry elements from a set of symmetry operations.
"""
import json
import numpy as np
from typing import List, Dict
from collections import defaultdict


def is_duplicate_vector(vec: np.ndarray, seen: List[np.ndarray], tol=1e-3) -> bool:
    """
    Check whether a vector (up to ± direction) already exists in seen.
    """
    for v in seen:
        if np.allclose(vec, v, atol=tol) or np.allclose(vec, -v, atol=tol):
            return True
    return False

def find_unique_elements(symmops: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """
    Classify unique symmetry elements and group them by type (e.g., "axis", "mirror").

    Args:
        symmops: List of 3×3 rotation matrices, 3×4 symmetry operation matrices or 4×4 affine matrices.

    Returns:
        A dictionary mapping each element type to a list of unique normalized vectors.
    """
    from symmetry_operations_interpreter import identify_rotation

    unique_elements: Dict[str, List[np.ndarray]] = defaultdict(list)

    for symmop in symmops:
        rotation = symmop[:3, :3]
        info = identify_rotation(rotation)

        direction = info.get("element")
        symm_type = info.get("rigid_op")

        if direction is None or symm_type is None:
            continue

        ele_type = "mirror" if symm_type == "m" else "axis"

        if not is_duplicate_vector(direction, unique_elements[ele_type]):
            unique_elements[ele_type].append(direction)

    return {
        element_type: [vec.tolist() for vec in vecs]
        for element_type, vecs in unique_elements.items()
    }


if __name__ == "__main__":
    with open("point_group_data/pg_u_ops.json", "r") as f:
        data = np.array(json.load(f))

    u_elements = find_unique_elements(list(data))

    with open("point_group_data/pg_u_elements.json", "w") as f:
        json.dump(u_elements, f, indent=4)

    print(f"Unique symmetry elements found: {len(u_elements['axis']) + len(u_elements['mirror'])}")
    print(f"Number of unique axes: {len(u_elements['axis'])}")
    print(f"Number of unique mirrors: {len(u_elements['mirror'])}")
