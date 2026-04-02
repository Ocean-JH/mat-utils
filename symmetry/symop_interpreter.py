#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2026-04-02
Description: Interpret the geometric meaning of symmetry operations in terms of point group operations.
"""
import numpy as np
import sympy as sp

TOL = 1e-6


def _parse_affine(affine_mat):
    """Return (rot, trans) from a 3x4 or 4x4 affine matrix."""
    affine_mat = np.asarray(affine_mat, dtype=float)
    if affine_mat.shape == (3, 4):
        return affine_mat[:, :3], affine_mat[:, 3]
    if affine_mat.shape == (4, 4):
        return affine_mat[:3, :3], affine_mat[:3, 3]
    raise ValueError(f"Affine matrix must be 3x4 or 4x4 shape, got {affine_mat.shape}.")


def is_close(a, b, tol=TOL):
    """Check if two scalars/arrays are close within tolerance."""
    return np.allclose(a, b, atol=tol, rtol=0.0)


def identify_rot_type(r_mat, tol=TOL):
    """
    Identify a 3×3 matrix as a point group operation.
    Source: ITC-Vol.A-2005, Table 11.2.1.1, p. 812
    """
    matrix = np.asarray(r_mat, dtype=float)
    det = np.linalg.det(matrix)
    trace = np.trace(matrix)

    if is_close(det, 1.0, tol):
        if is_close(trace, -1.0, tol):
            return "2"
        if is_close(trace, 0.0, tol):
            return "3"
        if is_close(trace, 1.0, tol):
            return "4"
        if is_close(trace, 2.0, tol):
            return "6"
        if is_close(trace, 3.0, tol):
            return "1"
        raise ValueError(f"Unrecognized rotation, det=1, trace={trace:.3f}")

    if is_close(det, -1.0, tol):
        if is_close(trace, -3.0, tol):
            return "-1"
        if is_close(trace, -2.0, tol):
            return "-6"
        if is_close(trace, -1.0, tol):
            return "-4"
        if is_close(trace, 0.0, tol):
            return "-3"
        if is_close(trace, 1.0, tol):
            return "m"
        raise ValueError(f"Unrecognized rotation, det=-1, trace={trace:.3f}")

    raise ValueError(f"Unrecognized rotation, det={det:.3f}, trace={trace:.3f}")

def format_sol_as_coords(sol_dict):
    """Format a sympy solution dict as a numeric 3D point string."""
    x, y, z = sp.symbols("x y z")
    coords = []
    for var in [x, y, z]:
        expr = sol_dict.get(var, var)
        if expr.is_Number:
            coords.append(f"{float(expr):.3f}")
        else:
            coords.append(str(expr))
    return ", ".join(coords)

def solve_inversion_point(rot, trans, tol=TOL):
    """Solve (W - I)x = -w for inversion point x."""
    rot = np.asarray(rot, dtype=float)
    trans = np.asarray(trans, dtype=float)
    A = rot - np.eye(3)
    b = -trans
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    if not is_close(A @ x, b, tol):
        raise ValueError("No stable solution for inversion point.")
    return x


def _solve_linear_fixed_set(W, t):
    """Solve (W - I)X = -t symbolically and return dict solutions."""
    x, y, z = sp.symbols("x y z")
    X = sp.Matrix([x, y, z])
    A = sp.Matrix(W) - sp.eye(3)
    b = -sp.Matrix(t)
    eqs = [(A * X - b)[i] for i in range(3)]
    return sp.solve(eqs, [x, y, z], dict=True)


def solve_rotoinversion_axis(rot, trans):
    """Solve fixed set of (W^2, Ww + w) for rotoinversion axis."""
    W2, t2 = affine_power(rot, trans, 2)
    return _solve_linear_fixed_set(W2, t2)[0]


def solve_screw_axis(rot, trans, n):
    """Return glide/screw component wg and axis/plane fixed set."""
    _, t_n = affine_power(rot, trans, n)
    w_g = sp.Matrix(t_n) / n
    w_l = sp.Matrix(trans) - w_g
    # Round w_l to mitigate numerical issues before symbolic solve
    sol = _solve_linear_fixed_set(rot, w_l.applyfunc(lambda x: round(float(x), 15)))[0]
    return w_g, sol


def rotation_order(rot, tol=TOL):
    """Determine the order of a rotation-like matrix."""
    rot = np.asarray(rot, dtype=float)
    det = np.linalg.det(rot)
    trace = np.trace(rot)
    cos_theta = (trace - 1.0) / 2.0

    if is_close(det, 1.0, tol):
        if is_close(cos_theta, 1.0, tol):
            return 1
        if is_close(cos_theta, 0.0, tol):
            return 4
        if is_close(cos_theta, -0.5, tol):
            return 3
        if is_close(cos_theta, -1.0, tol):
            return 2
        if is_close(cos_theta, 0.5, tol):
            return 6
        return 0

    if is_close(det, -1.0, tol):
        if is_close(cos_theta, -2.0, tol):
            return 1
        if is_close(cos_theta, -1.5, tol):
            return 6
        if is_close(cos_theta, -1.0, tol):
            return 4
        if is_close(cos_theta, -0.5, tol):
            return 3
        if is_close(cos_theta, 0.0, tol):
            return 2
        return 0

    return 0


def affine_power(rot, trans, n):
    """Compute (W, w)^n for an affine operation (W, w)."""
    rot = np.asarray(rot, dtype=float)
    trans = np.asarray(trans, dtype=float)

    y = np.zeros(3)
    Wk = np.eye(3)
    for _ in range(n):
        y += Wk @ trans
        Wk = Wk @ rot
    return Wk, y

def get_points_on_axis(sol_dict):
    """Convert one sympy solution dict to a numeric 3D point."""
    x, y, z = sp.symbols("x y z")
    exprs = [
        sol_dict.get(x, x),
        sol_dict.get(y, y),
        sol_dict.get(z, z)
    ]

    free_syms = set()
    for e in exprs:
        free_syms |= e.free_symbols

    param = list(free_syms)[0]

    # points = []
    # for i in range(3):
    #     expr = exprs[i]
    #     try:
    #         sol = sp.solve(expr, param)
    #         if sol:
    #             t_val = sol[0]
    #             subs = {param: t_val}
    #             pt = [float(e.subs(subs)) for e in exprs]
    #             points.append(np.array(pt))
    #     except:
    #         continue

    points = []
    for val in [0, 1]:
        subs = {param: val}
        pt = [float(e.subs(subs)) for e in exprs]
        points.append(np.array(pt))

    return points[0], points[1]


def _operation_sign_from_det(det_val, is_rotoinversion, tol=TOL):
    """Map determinant sign to sense label using ITC rule in prompt."""
    if abs(det_val) <= tol:
        raise ValueError("Determinant sign cannot be zero.")
    if not is_rotoinversion:
        return "+" if det_val > 0 else "-"
    return "+" if det_val < 0 else "-"


def get_operation_sense(rot, trans, tol=TOL):
    """
    Compute sense (+/-) for pure/screw rotation or rotoinversion using determinant rule.

    Returns:
        dict with keys:
            - rot_type: str
            - determinant: float
            - sense: '+', '-', or '0' (degenerate)
            - points: {'P0','P1','P2','P3'} for debugging/inspection
    """
    det = np.linalg.det(np.array(rot))
    n = rotation_order(rot, tol=tol)
    if n <= 0:
        raise ValueError("Cannot determine operation order for sense calculation.")

    # Axis from screw decomposition fixed set: (W - I)X = -w_l
    if det == -1:
        axis_sol = solve_rotoinversion_axis(rot, trans)
    else:
        w_g, axis_sol = solve_screw_axis(rot, trans, n)
    if not axis_sol:
        raise ValueError("No axis solution found for sense calculation.")

    # Pick first symbolic axis solution, evaluate two points with different parameter values
    P0, P1 = get_points_on_axis(axis_sol)
    if np.linalg.norm(P1 - P0) < tol:
        raise ValueError("Failed to construct two distinct axis points P0/P1.")
    # Choose a point not on axis
    axis_dir = P1 - P0
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    test_points = [
        P0 + np.array([0.0, 0.0, 0.0]),
        P0 + np.array([1.0, 0.0, 0.0]),
        P0 + np.array([0.0, 1.0, 0.0]),
        P0 + np.array([0.0, 0.0, 1.0]),
        P0 + np.array([1.0, 1.0, 0.0]),
        ]
    P2 = None
    for cand in test_points:
        if np.linalg.norm(np.cross(cand - P0, axis_dir)) > tol:
            P2 = cand
            break
    if P2 is None:
        raise ValueError("Failed to select point P2 outside axis.")

    # Apply affine op: P3 = W P2 + w
    P3 = rot @ P2 + trans

    v1 = P1 - P0
    v2 = P2 - P0
    v3 = P3 - P0
    d = float(np.linalg.det(np.column_stack([v1, v2, v3])))

    is_rotoinv = np.linalg.det(rot) < 0
    sense = _operation_sign_from_det(d, is_rotoinv, tol=tol)

    return sense

def identify_symmetry_operation(mat, tol=TOL):
    """
    Classify a 3x4 or 4x4 affine symmetry operation matrix.
    Returns concise text summary.
    """
    rot, trans = _parse_affine(mat)
    rot_type = identify_rot_type(rot, tol=tol)

    det = np.linalg.det(rot)
    trace = np.trace(rot)
    n = rotation_order(rot, tol=tol)

    w_g = ""
    glide_plane = ""
    rotoinversion_axis = ""
    screw_axis = ""
    inversion_point = ''
    rot_sense = ''

    if is_close(det, -1.0, tol):
        if is_close(trace, 1.0, tol):  # mirror/glide family
            w_g, glide_plane = solve_screw_axis(rot, trans, n)
            glide_plane = format_sol_as_coords(glide_plane)
        elif is_close(trace, -3.0, tol):  # rotoinversion family
            inversion_point = solve_inversion_point(rot, trans, tol=tol)
        else:
            inversion_point = solve_inversion_point(rot, trans, tol=tol)
            rotoinversion_axis = solve_rotoinversion_axis(rot, trans)
            rotoinversion_axis = format_sol_as_coords(rotoinversion_axis)
    elif is_close(det, 1.0, tol) and rot_type != "1":
        w_g, screw_axis = solve_screw_axis(rot, trans, n)
        screw_axis = format_sol_as_coords(screw_axis)

    elif rot_type == '1':
        pass

    else:
        raise ValueError("Unrecognized affine operation.")

    if rot_type not in ("1", "m", "-1", "2"):
        rot_sense = get_operation_sense(rot, trans, tol=tol)

    return f"{rot_type}{rot_sense} {list(w_g)} {glide_plane}{rotoinversion_axis}{screw_axis}   {inversion_point}"


########################################################################################################################
########################################################################################################################
########################################################################################################################
# The functions below has been deprecated!#
from fractions import Fraction
from numpy.linalg import norm


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
########################################################################################################################


if __name__ == '__main__':
    from pymatgen.symmetry.groups import PointGroup, SpaceGroup

    point_groups = [
        "1", "-1", "2", "m", "2/m",
        "222", "mm2", "mmm",
        "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
        "3", "-3", "32", "3m", "-3m",
        "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
        "23", "m-3", "432", "-43m", "m-3m"
    ]


    for pg in point_groups:
        pg_ops = PointGroup(pg)
        print(f"Point group {pg} has {len(pg_ops)} operations:")
        for op in pg_ops.symmetry_ops:
            result = identify_rot_type(op.rotation_matrix)
            print(f"Rotation: {op.rotation_matrix}\n Identification: {result}")

    for i in range(1, 231):
        sg = SpaceGroup.from_int_number(i)
        print(f"Space group {sg.symbol} (No. {i}) has {len(sg.symmetry_ops)} operations:")
        for op in sg.symmetry_ops:
            result = identify_symmetry_operation(op.affine_matrix)
            print(f"Affine matrix:\n{op.affine_matrix}\n Identification:\n{result}\n")