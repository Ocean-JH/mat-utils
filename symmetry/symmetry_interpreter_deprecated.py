#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-07-01
Description: 
"""
# The functions in this module are deprecated. Use `symmetry_interpreter.py` instead.
import numpy as np
from fractions import Fraction
from numpy.linalg import norm
import warnings

TOL=1e-6

def normalize_vec(vec, tol=TOL, max_denominator=10):
    """
    Normalize and canonicalize a vector to smallest integer representation.
    Example: [-0.44721, 0.89443, 0] -> [1, -2, 0]
    """
    vec = np.asarray(vec, dtype=float)
    if norm(vec) < tol:
        return np.array([0, 0, 1])    # Default axis

    # Normalization: Ensure elements are integers within [-max_abs, max_abs]
    vec /= norm(vec)

    vec_scaled = [Fraction(v).limit_denominator(max_denominator) for v in vec]
    denominators = [f.denominator for f in vec_scaled if isinstance(f, Fraction)]
    lcm = np.lcm.reduce(denominators) if denominators else 1

    int_vec = np.array([
        int(round(f * lcm)) if isinstance(f, Fraction) else 0
        for f in vec_scaled
    ])

    gcd = np.gcd.reduce(np.abs(int_vec[int_vec != 0])) if np.any(int_vec != 0) else 1
    int_vec = int_vec // gcd if gcd != 0 else int_vec

    # Canonicalization: ensure first non-zero element is positive
    for i in range(len(int_vec)):
        if abs(int_vec[i]) > 0:
            if int_vec[i] < 0:
                int_vec = - int_vec
            break

    return int_vec

def rot_order(r_mat, max_order=6, tol=TOL):
    """Return the minimal integer n such that mat^n = I."""
    product = np.identity(3)
    for n in range(1, max_order + 1):
        product = product @ r_mat
        if np.allclose(product, np.identity(3), atol=tol):
            return n
    return None

# The result of this function is not always reliable, Use with caution.
def rot_direction(matrix, axis, tol=TOL):
    """
    Use triple product to determine rotation direction.
    For 180° rotations (C2, S2), the direction is undefined → return '' (empty).

    Parameters:
        matrix (np.ndarray): 3×3 rotation or rotoinversion matrix.
        axis (np.ndarray): Normalized axis vector.
        tol (float): Tolerance.

    Returns:
        str: '⁺', '⁻', or '' for undefined (e.g. C2/S2)
    """
    # Check if it's a 180° rotation
    angle = np.arccos((np.trace(matrix) - 1) / 2)
    angle_deg = np.degrees(angle) % 360
    if np.isclose(angle_deg, 180.0, atol=1e-3):
        return ''

    # Choose a vector orthogonal to axis
    candidates = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    for v in candidates:
        if norm(np.cross(axis, v)) > tol:
            break
    # Project v to plane orthogonal to axis
    v_proj = v - np.dot(v, axis) * axis
    v_proj = normalize_vec(v_proj, tol)
    v_rot = matrix @ v_proj
    triple = np.dot(axis, np.cross(v_proj, v_rot))
    return "⁺" if triple < 0 else "⁻"

def identify_rotation(r_mat, tol=TOL):
    """
    Identify a 3×3 matrix as a point group operation.

    Parameters:
        r_mat (array-like): A 3x3 rotation-like matrix from symmetry operation.
        tol (float): Numerical tolerance for equality checks.

    Returns:
        dict: {
            'rigid_op': str                # Type of rigid operations. e.g., 'C3⁺', 'm', 'i', 'S4⁺'
            'element': array or None,       # axis (for rotation) or normal (for mirror)
            'angle_deg': float or None,     # Rotation angle in degrees
        }
    """
    import warnings
    warnings.warn("This function is deprecated. Use `identify_rot_type` instead.", DeprecationWarning)

    matrix = np.array(r_mat, dtype=float)
    det = np.linalg.det(matrix)
    trace = np.trace(matrix)

    # Identity
    if np.allclose(matrix, np.eye(3), atol=tol):
        return {"rigid_op": "E", "element": None, "angle_deg": 0.0}

    # Inversion (−I)
    if np.allclose(matrix, - np.eye(3), atol=tol):
        return {"rigid_op": "i", "element": None, "angle_deg": 180.0}

    # Mirror plane (det = -1, trace = 1)
    if np.isclose(det, -1.0, atol=tol) and np.isclose(trace, 1.0, atol=tol):
        eigvals, eigvecs = np.linalg.eig(matrix)
        mirror_indices = np.where(np.isclose(np.real(eigvals), -1.0, atol=tol))[0]
        if len(mirror_indices) == 1:
            mirror = np.real(eigvecs[:, mirror_indices[0]])
            mirror = normalize_vec(mirror, tol=tol)

            return {
                "rigid_op": "m",
                "element": mirror,
                "angle_deg": 180.0
            }
        else:
            raise ValueError("⚠️ Mirror plane ambiguous or not found.")


    # Proper rotation (det = +1)
    if np.isclose(det, 1.0, atol=tol):
        n = rot_order(matrix, max_order=6, tol=tol)
        eigvals, eigvecs = np.linalg.eig(matrix)
        axis_indices = np.where(np.isclose(np.real(eigvals), 1.0, atol=tol))[0]
        if len(axis_indices) == 1:
            axis = np.real(eigvecs[:, axis_indices[0]])
            axis = normalize_vec(axis, tol)
        else:
            axis = None

        angle_raw = 360.0 / n if n else None
        direction = rot_direction(matrix, axis) if axis is not None else "?"

        if angle_raw is not None:
            if direction == "⁻":
                angle_deg = (360 - angle_raw) % 360
            else:
                angle_deg = angle_raw % 360

        label = f"C{n}{direction}" if n else "Cn"

        return {"rigid_op": label, "element": axis, "angle_deg": angle_deg}

    # Imporper Rotation: Rotation + Inversion (det = -1)
    if np.isclose(det, -1.0, atol=tol):
        minus_mat = -matrix
        n = rot_order(minus_mat, max_order=6, tol=tol)
        eigvals, eigvecs = np.linalg.eig(minus_mat)
        axis_indices = np.where(np.isclose(np.real(eigvals), 1.0, atol=tol))[0]
        if len(axis_indices) == 1:
            axis = np.real(eigvecs[:, axis_indices[0]])
            axis = normalize_vec(axis, tol)
        else:
            axis = None

        angle_raw = 360.0 / n if n else None
        direction = rot_direction(minus_mat, axis) if axis is not None else "?"

        if angle_raw is not None:
            if direction == "⁻":
                angle_deg = (360 - angle_raw) % 360
            else:
                angle_deg = angle_raw % 360

        label = f"S{n}{direction}" if n else "Sn"

        return {"rigid_op": label, "element": axis, "angle_deg": angle_deg}

    return {
        "rigid_op": "Unknown",
        "element": None,
        "angle_deg": None
    }


if __name__ == "__main__":
    warnings.warn("This module is deprecated. Use `symmetry_interpreter.py` instead.", DeprecationWarning)