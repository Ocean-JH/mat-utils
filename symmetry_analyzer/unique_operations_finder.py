#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-07-08
Description: Find unique symmetry operations from space group data.
"""
import re
import json
import numpy as np

# Function to convert operation string to affine transformation matrix
def op_str_to_affine(op_str, output="full"):
    """
    Convert a symmetry operation string to an affine transformation matrix.
    """
    rot_match = re.search(r"Rot:\n\[\[(.*?)\]\]", op_str, re.DOTALL)
    tau_match = re.search(r"tau\n\[(.*?)\]", op_str, re.DOTALL)

    if not rot_match or not tau_match:
        print("error: Invalid operation string format")
        return None

    rot_str = rot_match.group(1)
    rot_rows = rot_str.strip().split("]\n [")
    rot_matrix = np.array([[float(x) for x in row.strip().split()] for row in rot_rows])

    tau_str = tau_match.group(1)
    tau_vector = np.array([float(x) for x in tau_str.strip().split()])

    if rot_matrix.shape != (3, 3) or tau_vector.shape != (3,):
        raise Exception("Invalid shape of rotation matrix")

    if output == "rot":
        return rot_matrix.tolist()
    elif output == "tau":
        return tau_vector.tolist()
    elif output == "full":
        affine = np.eye(4)
        affine[:3, :3] = rot_matrix
        affine[:3, 3] = tau_vector
        return affine.tolist()
    else:
        raise ValueError("Invalid output type specified. Use 'rot', 'tau', or 'full'.")


def deduplicate_operations(data, output="full"):
    """Deduplicate operations from space group data."""
    unique_ops = []
    seen = []

    for sg_data in data.values():
        # for op_str in sg_data.get("operations", []):
        for i in range(len(sg_data)):
            affine = op_str_to_affine(sg_data[i], output=output)
            if affine is None:
                print(f"error: Invalid operation string format in {sg_data[i]}")
                continue

            already_seen = any(np.allclose(affine, other) for other in seen)
            if not already_seen:
                seen.append(affine)
                unique_ops.append(affine)

    return unique_ops


if __name__ == "__main__":
    with open("point_group_data/symmops_by_pg.json") as f:
        spacegroup_data = json.load(f)

    print("Find", len(spacegroup_data), "Point groups in the data.")
    unique_operations = deduplicate_operations(spacegroup_data, output="full")

    print("There are", len(unique_operations), "unique operations found.")

    with open("point_group_data/pg_u_ops.json", "w") as f:
        json.dump(unique_operations, f, indent=2)
