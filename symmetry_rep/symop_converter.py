#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-07-12
Description: This script provides functions to parse symmetry operation strings, convert them to rotation matrices and translation vectors, and construct affine matrices. It also includes functionality to convert back from rotation matrices and translation vectors to symmetry operation strings.
"""
import re
import numpy as np
from fractions import Fraction

_AXES = ('x', 'y', 'z')

def parse_symop(symop: str):
    """Parse a symmetry operation string into rotation matrix R and translation vector t."""
    exprs = [e.strip() for e in symop.strip().lstrip('(').rstrip(')').split(',')]
    if len(exprs) != 3:
        raise ValueError("Symmetry operation must contain 3 coordinate expressions.")

    R = np.zeros((3, 3), dtype=int)
    t = np.zeros(3, dtype=float)

    for i, expr in enumerate(exprs):
        for j, ax in enumerate(_AXES):
            m = re.search(rf'([+-]?)\b{ax}\b', expr)
            if m:
                sign = -1 if m.group(1) == '-' else 1
                R[i, j] = sign
                expr = re.sub(rf'([+-]?)\b{ax}\b', '', expr, count=1)
        expr = expr.replace(' ', '')
        if expr:
            t[i] = float(Fraction(expr))
    return R, t % 1.0  # Ensure translation is within [0, 1)

def to_affine(R, t):
    """Construct a 4×4 homogeneous affine matrix from R and t."""
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = t
    return A

def symop_string_from_matrix(R, t, tol=1e-8):
    """Convert rotation matrix R and translation vector t to a symmetry operation string."""
    exprs = []
    for i in range(3):
        terms = []
        for j in range(3):
            coef = R[i, j]
            if coef == 1:
                terms.append(f"+{_AXES[j]}")
            elif coef == -1:
                terms.append(f"-{_AXES[j]}")
        shift = t[i] % 1
        if abs(shift) > tol:
            frac = Fraction(shift).limit_denominator(12)
            terms.append(f"+{frac}" if frac > 0 else f"{frac}")
        # Join terms and remove leading '+'
        expr = ''.join(terms)
        exprs.append(expr if not expr.startswith('+') else expr[1:])
    return '(' + ', '.join(exprs) + ')'


if __name__ == "__main__":
    # Example usage
    symop = "-x+y,y,-z+1/2"
    R, t = parse_symop(symop)
    print("Rotation Matrix R:\n", R)
    print("Translation Vector t:", t)

    A = to_affine(R, t)
    print("Affine Matrix A:\n", A)

    symop_str = symop_string_from_matrix(R, t)
    print("Symmetry Operation String:", symop_str)
