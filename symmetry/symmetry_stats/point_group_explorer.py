#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-xx-xx
Description: [Brief description of the script's purpose]
"""
from pymatgen.symmetry.groups import PointGroup


def point_group_symmetry_operations(pg_symbol: str):
    """
    Print the symmetry operations of a given point group.
    Args:
        pg_symbol (str): The name of the point group (e.g., "m-3m", "4/mmm").
    """
    try:
        pg = PointGroup(pg_symbol)
    except Exception as e:
        print(f"Can not recongnize point group '{pg_symbol}': {e}")
        return

    symm_ops = [str(op) for op in pg.symmetry_ops]

    return list(symm_ops)

    # print(f"Point Group: {pg_symbol}  contains {len(symm_ops)} symmetry operations.")
    # for i, op in enumerate(symm_ops, 1):
    #     print(f"Operation {i}:")
    #     print(f"    matrix:\n{op.rotation_matrix}")
    #     print(f"    translation vector: {op.translation_vector}")
    #     print("-" * 30)


if __name__ == "__main__":
    point_groups = [
        "1", "-1", "2", "m", "2/m",
        "222", "mm2", "mmm",
        "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
        "3", "-3", "32", "3m", "-3m",
        "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
        "23", "m-3", "432", "-43m", "m-3m"
    ]

    symmop = PointGroup("m-3m")
    for i, op in enumerate(symmop):
        print("Operation {}:".format(i + 1))
        print(op.rotation_matrix)
        print("-" * 30)

    # symmops = {}
    # for pg_name in point_groups:
    #     symmop = point_group_symmetry_operations(pg_name)
    #     symmops[pg_name] = symmop
    #
    # with open("symmops_by_pg.json", "w") as f:
    #     json.dump(symmops, f, indent=4)
