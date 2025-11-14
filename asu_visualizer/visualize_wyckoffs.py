#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-11-12
Description:
"""
import os
import json
from fractions import Fraction
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from plotly import graph_objs as go
from pyxtal.symmetry import Group

from exact_cuts import get_asu

EPS = 1e-7
space_group_bases: np.ndarray = np.load("data/space_group_basis_ndarrays.npz")["bases"]  # (230,3,3)


# ---------- Helpers ----------
def generate_monochromatic_palette(num_colors: int, base_hue: float = 0.12) -> List[str]:
    import colorsys
    out = []
    for i in range(num_colors):
        light = 0.35 + (i / max(1, num_colors - 1)) * 0.5
        r, g, b = colorsys.hls_to_rgb(base_hue, light, 0.38)
        out.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
    return out


def generate_muted_rainbow_colors(num_colors: int, saturation=0.55, value=0.8) -> List[str]:
    import colorsys
    cols = []
    for i in range(num_colors):
        h = i / num_colors
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        cols.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
    return cols


def get_hall_number(space_group_number) -> int:
    """
    Convert space group number to Hall number.

    Parameters:
    space_group_number: The space group number (1-230).

    Returns:
    int: The corresponding Hall number.
    """

    with open('data/std_spg_HM_map.json', 'r') as f:
        space_group_to_hall = json.load(f)

    return space_group_to_hall.get(str(space_group_number), None)


def frac_array(seq) -> np.ndarray:
    vfunc = np.vectorize(lambda s: float(Fraction(s)))
    return vfunc(seq)


def rational_to_arr(val):
    """
    Convert a scalar or array-like of rational/int/float values to float(s).
    """
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.array([rational_to_arr(v) for v in val], dtype=float)
    try:
        return float(val)
    except Exception:
        try:
            return float(val.numerator() / val.denominator())
        except AttributeError:
            raise TypeError(f"Unsupported type for conversion: {type(val)}")


def inside_unit_cell(x: np.ndarray) -> np.ndarray:
    return np.all((x >= -EPS) & (x <= 1 + EPS), axis=-1)


def apply_pyxtal_op(points: np.ndarray, op) -> np.ndarray:
    return points @ op.rotation_matrix.T + op.translation_vector.reshape(1, 3)


# ---------- Core visualization ----------
def visualize_wyckoffs_plotly(
    space_group_number: int,
    plot_asu: bool = True,
    exact_asu: bool = False,
    plot_wyckoffs: bool = True,
    orbit_wyckoffs: bool = True,
    plot_unit_cell: bool = True,
    theme: str = "light",
    camera: Optional[Dict[str, Any]] = None,
    html_out: Optional[str] = None,
    image_out: Optional[str] = None,
    image_scale: int = 2,
):
    hall_number = get_hall_number(space_group_number)
    sg = Group(hall_number, use_hall=True)
    general_ops = sg.Wyckoff_positions[0]
    basis = space_group_bases[space_group_number - 1]  # (3,3)

    # Load wyckoff position data
    with open('data/wyckoffs_in_asu.json') as f:
        wyckoff_db = json.load(f)
    wyckoffs = wyckoff_db[str(space_group_number)]

    translations = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0],
        [0, 0, -1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [-1, 1, 0],
        [1, -1, 0], [-1, -1, 0], [-1, 0, 1], [1, 0, -1],
        [-1, 0, -1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
        [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]
    ], dtype=float)

    # Traces accumulator
    traces = []
    category_indices: Dict[str, List[int]] = {
        "asu": [], "cell": [], "0D": [], "1D": [], "2D": []
    }

    # ---------- Unit cell edges ----------
    if plot_unit_cell:
        # original sequence of segment endpoint pairs (start, end, start, end, ...)
        raw_points = np.array([
            [0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1],
            [1,0,0],[1,1,0],[1,0,0],[1,0,1],[0,1,0],[1,1,0],
            [0,1,0],[0,1,1],[0,0,1],[1,0,1],[0,0,1],[0,1,1],
            [1,1,1],[0,1,1],[1,1,1],[1,0,1],[1,1,1],[1,1,0]
        ], dtype=float)
        # Insert NaN separators between each pair so Plotly does not draw connecting lines
        nan = np.array([np.nan, np.nan, np.nan], dtype=float)
        segs = []
        for i in range(0, raw_points.shape[0], 2):
            segs.append(raw_points[i])
            segs.append(raw_points[i+1])
            segs.append(nan)
        edges = np.array(segs) @ basis.T
        traces.append(go.Scatter3d(
            x=edges[:,0], y=edges[:,1], z=edges[:,2],
            mode="lines",
            line=dict(color="black", width=4),
            name="Unit Cell",
            hoverinfo="skip",
            showlegend=True
        ))
        category_indices["cell"].append(len(traces) - 1)

    # ---------- ASU ----------
    if plot_asu:
        asu_vertices = rational_to_arr(
            get_asu(space_group_number).shape_vertices(include_open_vertices=True)
        ) @ basis.T
        if not exact_asu:
            hull = ConvexHull(asu_vertices)
            i, j, k = hull.simplices[:,0], hull.simplices[:,1], hull.simplices[:,2]
            traces.append(go.Mesh3d(
                x=asu_vertices[:,0], y=asu_vertices[:,1], z=asu_vertices[:,2],
                i=i, j=j, k=k,
                name="ASU (approx)",
                color="orange",
                opacity=0.5,
                flatshading=True,
            ))
            category_indices["asu"].append(len(traces) - 1)
            # Optional edges overlay
            # Build edge segments with NaN separators so Plotly won't join distinct segments
            edge_pts_with_seps = []
            nan = np.array([np.nan, np.nan, np.nan], dtype=float)
            for tri in hull.simplices:
                for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    p1 = asu_vertices[a]; p2 = asu_vertices[b]
                    edge_pts_with_seps.append(p1)
                    edge_pts_with_seps.append(p2)
                    edge_pts_with_seps.append(nan)
            edge_segments = np.array(edge_pts_with_seps)
            traces.append(go.Scatter3d(
                x=edge_segments[:,0], y=edge_segments[:,1], z=edge_segments[:,2],
                mode="lines",
                line=dict(color="red", width=8),
                name="ASU edges",
                hoverinfo="skip",
                showlegend=False
            ))
            category_indices["asu"].append(len(traces) - 1)
        else:
            # Exact ASU faces/edges from JSON
            with open('data/exact_asu_edges_and_faces.json') as f:
                exact_data = json.load(f)[str(space_group_number)]
            # Faces
            face_points_all = []
            tri_i = []
            tri_j = []
            tri_k = []
            offset = 0
            for face in exact_data["faces"]:
                for poly in face:
                    poly_arr = frac_array(poly).astype(float)
                    poly_arr = poly_arr @ basis.T
                    face_points_all.append(poly_arr)
                    # Fan triangulation
                    for t in range(1, len(poly_arr)-1):
                        tri_i.append(offset)
                        tri_j.append(offset + t)
                        tri_k.append(offset + t + 1)
                    offset += len(poly_arr)
            if face_points_all:
                all_pts = np.concatenate(face_points_all, axis=0)
                traces.append(go.Mesh3d(
                    x=all_pts[:,0], y=all_pts[:,1], z=all_pts[:,2],
                    i=tri_i, j=tri_j, k=tri_k,
                    name="ASU (exact)",
                    color="orange",
                    opacity=0.5,
                ))
                category_indices["asu"].append(len(traces) - 1)
            # Edges
            edge_pts = []
            for edge in exact_data["edges"]:
                for seg in edge:
                    p1 = np.array([float(Fraction(u)) for u in seg[0]])
                    p2 = np.array([float(Fraction(u)) for u in seg[1]])
                    # keep raw fractional points for clean separators, transform later
                    edge_pts.append(p1)
                    edge_pts.append(p2)
                    edge_pts.append(np.array([np.nan, np.nan, np.nan], dtype=float))
            if edge_pts:
                edge_pts = np.array(edge_pts) @ basis.T
                traces.append(go.Scatter3d(
                    x=edge_pts[:,0], y=edge_pts[:,1], z=edge_pts[:,2],
                    mode="lines",
                    line=dict(color="red", width=8),
                    name="ASU edges",
                    hoverinfo="skip",
                    showlegend=False
                ))
                category_indices["asu"].append(len(traces) - 1)

    if plot_wyckoffs:
        # Palettes
        colors_0d = generate_monochromatic_palette(len(wyckoffs["0D"]), base_hue=0.08)
        colors_1d = generate_monochromatic_palette(len(wyckoffs["1D"]), base_hue=0.55)
        colors_2d = generate_muted_rainbow_colors(len(wyckoffs["2D"]), saturation=0.5, value=0.85)

        # 0D
        for idx, (label, pt) in enumerate(wyckoffs["0D"].items()):
            arr = frac_array(pt).astype(float)  # (1,3)
            images = []
            if orbit_wyckoffs:
                for op in general_ops:
                    eq = apply_pyxtal_op(arr, op)
                    for t in translations:
                        cand = eq + t
                        if inside_unit_cell(cand):
                            images.append(cand)
            else:
                images.append(arr)
            if images:
                pts = np.concatenate(images, axis=0) @ basis.T
                traces.append(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    mode="markers",
                    marker=dict(size=10, color=colors_0d[idx], opacity=1),
                    name=f"0D {label}",
                    hovertemplate=f"Wyckoff {label}<br>x=%{{x:.3f}} y=%{{y:.3f}} z=%{{z:.3f}}<extra></extra>"
                ))
                category_indices["0D"].append(len(traces) - 1)

        # 1D
        for idx, (label, segments) in enumerate(wyckoffs["1D"].items()):
            segs = frac_array(segments).astype(float)  # (n,2,3)
            collected = []
            if orbit_wyckoffs:
                for seg in segs:
                    for op in general_ops:
                        eq = apply_pyxtal_op(seg, op)
                        for t in translations:
                            cand = eq + t
                            if np.all(inside_unit_cell(cand)):
                                collected.append(cand)
            else:
                collected.extend(segs)
            if collected:
                # Insert NaN separators between segments so Plotly does NOT
                # draw connecting lines between different segments.
                pts_with_seps = []
                nan_row = np.array([[np.nan, np.nan, np.nan]], dtype=float)
                for seg in collected:
                    # seg is (2,3) or (m,3) for a poly-segment; keep as-is then add separator
                    pts_with_seps.append(seg)
                    pts_with_seps.append(nan_row)
                pts_arr = np.concatenate(pts_with_seps, axis=0)  # (N,3) with NaN rows
                pts = pts_arr @ basis.T
                traces.append(go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    mode="lines",
                    line=dict(color=colors_1d[idx], width=8),
                    name=f"1D {label}",
                    hoverinfo="skip",
                ))
                category_indices["1D"].append(len(traces) - 1)

        # 2D
        for idx, (label, facets) in enumerate(wyckoffs["2D"].items()):
            face_pts = []
            tri_i = []
            tri_j = []
            tri_k = []
            offset = 0
            for facet in facets:
                for poly in facet:
                    poly_arr = frac_array(poly).astype(float)  # (m,3)
                    if orbit_wyckoffs:
                        for op in general_ops:
                            transformed = apply_pyxtal_op(poly_arr, op)
                            for t in translations:
                                cand = transformed + t
                                if np.all(inside_unit_cell(cand)):
                                    face_pts.append(cand @ basis.T)
                                    for tix in range(1, cand.shape[0]-1):
                                        tri_i.append(offset)
                                        tri_j.append(offset + tix)
                                        tri_k.append(offset + tix + 1)
                                    offset += cand.shape[0]
                    else:
                        face_pts.append(poly_arr @ basis.T)
                        for tix in range(1, poly_arr.shape[0]-1):
                            tri_i.append(offset)
                            tri_j.append(offset + tix)
                            tri_k.append(offset + tix + 1)
                        offset += poly_arr.shape[0]
            if face_pts:
                all_pts = np.concatenate(face_pts, axis=0)
                traces.append(go.Mesh3d(
                    x=all_pts[:,0], y=all_pts[:,1], z=all_pts[:,2],
                    i=tri_i, j=tri_j, k=tri_k,
                    name=f"2D {label}",
                    color=colors_2d[idx],
                    opacity=0.55,
                    flatshading=True,
                ))
                category_indices["2D"].append(len(traces) - 1)

    # ---------- Build visibility toggles ----------
    def make_visibility_mask(active_keys: List[str]) -> List[bool]:
        vis = [False] * len(traces)
        for key in active_keys:
            for idx in category_indices[key]:
                vis[idx] = True
        return vis

    buttons = [
        dict(label="Show All",
             method="update",
             args=[{"visible": make_visibility_mask(["asu","cell","0D","1D","2D"])}]),
        dict(label="ASU View",
             method="update",
             args=[{"visible": make_visibility_mask(["asu","cell","0D","1D"])}]),
        dict(label="ASU only",
             method="update",
             args=[{"visible": make_visibility_mask(["asu"])}]),
        dict(label="Unit Cell only",
             method="update",
             args=[{"visible": make_visibility_mask(["cell"])}]),
        dict(label="0D",
             method="update",
             args=[{"visible": make_visibility_mask(["0D", "cell"])}]),
        dict(label="1D",
             method="update",
             args=[{"visible": make_visibility_mask(["1D", "cell"])}]),
        dict(label="2D",
             method="update",
             args=[{"visible": make_visibility_mask(["2D", "cell"])}]),
        dict(label="Hide All",
             method="update",
             args=[{"visible": make_visibility_mask([])}]),
    ]

    # Initial visibility: show all requested
    initial_visible = make_visibility_mask(
        [k for k in ["asu","cell","0D","1D"] if category_indices[k]]
    )
    for i, v in enumerate(initial_visible):
        traces[i].visible = v

    # ---------- Layout ----------
    if camera is None:
        camera = dict(
            eye=dict(x=1.6, y=1.4, z=1.2),
            center=dict(x=0.0, y=0.0, z=0.0),
        )

    bg_color = "#ffffff" if theme == "light" else "#111418"
    paper_color = "#ffffff" if theme == "light" else "#0c0f12"
    font_color = "#222" if theme == "light" else "#e2e2e2"
    legend_bg = "rgba(255,255,255,0.6)" if theme == "light" else "rgba(40,40,40,0.6)"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Space group {space_group_number}",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=camera,
        ),
        showlegend=True,
        legend=dict(bgcolor=legend_bg, font=dict(color=font_color, size=12)),
        paper_bgcolor=paper_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color, family="Helvetica"),
        margin=dict(l=0, r=0, b=0, t=50),
        updatemenus=[
            dict(
                type="buttons",
                buttons=buttons,
                direction="down",
                x=0.02, y=0.95,
                pad=dict(r=4, t=4),
                showactive=True,
                bgcolor=legend_bg,
                bordercolor="#888",
                font=dict(color=font_color, size=12),
            )
        ]
    )

    # Optional exports
    if html_out:
        fig.write_html(f"ASU_of_SG_{space_group_number}.html", include_mathjax='cdn')
    if image_out:
        try:
            fig.write_image(f"ASU_of_SG_{space_group_number}.png", scale=image_scale)
        except ValueError:
            print("Static image export requires kaleido: pip install -U kaleido")

    # Show interactive window
    fig.show()
    return fig


if __name__ == "__main__":
    # Example usage: adjust parameters below.
    visualize_wyckoffs_plotly(
        space_group_number=192,
        plot_asu=True,
        exact_asu=True,
        plot_wyckoffs=True,
        orbit_wyckoffs=True,
        plot_unit_cell=True,
        theme="light",          # or "dark"
        camera=dict(eye=dict(x=1.8, y=1.2, z=1.3))
    )
