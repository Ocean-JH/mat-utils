#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-12-06
Description:
    Visualize space-group relations (super/sub) as tree graphs.
"""
from __future__ import annotations

import os
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx

from pymatgen.core.composition import Composition
from mp_api.client import MPRester

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_ORDER_PATH = DATA_DIR / "sg_to_order.json"
DEFAULT_SUPER_PATH = DATA_DIR / "supergroup.json"
DEFAULT_SUB_PATH = DATA_DIR / "subgroup.json"


def load_database(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def normalize_formula(formula: str) -> str:
    """Normalize a chemical formula to a reduced form for comparison."""
    try:
        comp = Composition(formula)
        return comp.reduced_formula
    except Exception:
        return formula.replace(" ", "")

def query_entries(
        formula: str,
        api_key: Optional[str] = None,
        save_structures: bool = False
):
    path = Path(DATA_DIR / f"mp-{formula}.json")
    api_key = api_key or os.environ.get("MP_API_KEY")

    if not api_key:
        raise ValueError("API key is required to fetch data from Materials Project.")

    fields = [
        "material_id",
        "formula_pretty",
        "symmetry",
        "energy_above_hull",
        "formation_energy_per_atom",
        "is_stable",
        "theoretical",
        "database_IDs",
    ]

    if save_structures:
        fields.append("structure")

    with MPRester(api_key) as mpr:
        print(mpr.materials.summary.available_fields)
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=fields,
        )

    if docs:
        entries: List[Dict[str, Any]] = []
        for doc in docs:
            entry = {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "spacegroup": {
                    "number": doc.symmetry.number,
                    "symbol": doc.symmetry.symbol,
                    "point_group": doc.symmetry.point_group,
                },
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
                "is_stable": doc.is_stable,
                "experimentally_verified": (not doc.theoretical) and bool(doc.database_IDs),
                "database_IDs": doc.database_IDs
            }
            entries.append(entry)

        with path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)
    else:
        raise ValueError(f"No entries found for formula: {formula}")

    if save_structures:
        struct_path = DATA_DIR / f"mp-{formula}-structures"
        struct_path.mkdir(exist_ok=True)
        for doc in docs:
            struct_file = struct_path / f"{doc.material_id}.cif"
            with struct_file.open("w", encoding="utf-8") as sfh:
                sfh.write(doc.structure.to(fmt="cif"))

    return entries

def load_entries(formula: str, api_key: Optional[str] = None, save_structures: bool = False) -> List[Dict[str, Any]]:
    path = Path(DATA_DIR / f"mp-{formula}.json")

    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)
    else:
        entries = query_entries(formula, api_key=api_key, save_structures=save_structures)

    return entries

def create_metadata(entries: List[Dict]) -> Dict[str, Any]:
    """Build metadata dictionary from entries."""
    metadata: Dict[str, List[Dict[str, Any]]] = {}

    for entry in entries:
        sg = entry["spacegroup"].get("number")
        sg_key = str(int(sg))

        if sg is None:
            continue

        meta = {
            "material_id": entry.get("material_id"),
            "formula": entry.get("formula"),
            "formation_energy_per_atom": entry.get("formation_energy_per_atom"),
            "energy_above_hull": entry.get("energy_above_hull"),
            "is_stable": entry.get("is_stable"),
            "experimentally_verified": entry.get("experimentally_verified"),
            "database_IDs": entry.get("database_IDs")
        }

        metadata.setdefault(sg_key, []).append(meta)

    for entries in metadata.values():
        entries.sort(
            key=lambda m: (
                m["energy_above_hull"] is None,
                m["energy_above_hull"] if m["energy_above_hull"] is not None else float("inf")
            )
        )

    return metadata


@dataclass
class GroupRelations:
    orders: Dict[str, int] = field(default_factory=dict)
    super_rel: Dict[str, Dict] = field(default_factory=dict)
    sub_rel: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)

    @classmethod
    def from_files(cls,
                   order_path: Path = DEFAULT_ORDER_PATH,
                   super_path: Path = DEFAULT_SUPER_PATH,
                   sub_path: Path = DEFAULT_SUB_PATH) -> "GroupRelations":
        return cls(
            orders=load_database(order_path),
            super_rel=load_database(super_path),
            sub_rel=load_database(sub_path),
        )

    def get_symbol(self, group: str) -> str:
        entry = self.super_rel.get(str(group))
        return entry.get("HMsymbol", "?") if entry else "?"

    def get_order(self, group: str) -> int:
        return int(self.orders.get(str(group), 0))

    def _neighbors(self,
                   group: str,
                   relation_type: str,
                   index_bounds: Optional[Tuple[Optional[int], Optional[int]]] = None
                   ) -> List[Tuple[str, List[int], List[int]]]:
        group = str(group)
        relation_type = relation_type.lower()
        if relation_type == "supergroup":
            raw = self.super_rel.get(group, {}).get("supergroups", {})
        elif relation_type == "subgroup":
            raw = self.sub_rel.get(group, {})
        else:
            raise ValueError(f"Relation type must be 'supergroup' or 'subgroup', got: {relation_type}")

        if index_bounds is None:
            credible_index, visible_index = 2, None
        else:
            credible_index, visible_index = index_bounds
        results: List[Tuple[str, List[int], List[int]]] = []
        for neighbor, indices in raw.items():
            if neighbor == group:
                continue
            cred_idx = [
                idx for idx in indices
                if credible_index is None or idx < credible_index
            ]
            vis_idx = [
                idx for idx in indices
                if visible_index is None or idx < visible_index
            ]
            if not cred_idx and not vis_idx:
                continue
            results.append((str(neighbor), cred_idx, vis_idx))
        return results

    def _traverse_direction(self,
                            root: str,
                            relation_type: str,
                            index_bounds: Optional[Tuple[Optional[int], Optional[int]]] = None
                            ) -> Tuple[Set[str], Dict[Tuple[str, str], Dict[str, Set[int]]]]:
        visited: Set[str] = {root}
        edges: Dict[Tuple[str, str], Dict[str, Set[int]]] = {}
        queue: deque[str] = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor, cred_idx, vis_idx in self._neighbors(current, relation_type, index_bounds):
                edge = (current, neighbor)
                data = edges.setdefault(edge, {"credible": set(), "visible": set()})
                data["credible"].update(cred_idx)
                data["visible"].update(vis_idx)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited, edges

    def _level_map(self, nodes: Set[str]) -> Dict[str, int]:
        unique_orders = sorted({self.get_order(n) for n in nodes}, reverse=True)
        order_to_level = {order: idx for idx, order in enumerate(unique_orders)}
        return {node: order_to_level[self.get_order(node)] for node in nodes}

    def _has_step_path(
        self,
        src: str,
        dst: str,
        direction: int,
        level_map: Dict[str, int],
        adjacency: Dict[str, Set[str]],
        exclude: Tuple[str, str],
    ) -> bool:
        target_level = level_map[dst]
        queue: deque[str] = deque([src])
        visited: Set[str] = {src}
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                if (node, neighbor) == exclude:
                    continue
                if level_map.get(neighbor) is None:
                    continue
                if level_map[neighbor] - level_map[node] != direction:
                    continue
                if direction > 0 and level_map[neighbor] > target_level:
                    continue
                if direction < 0 and level_map[neighbor] < target_level:
                    continue
                if neighbor == dst:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def _remove_cross_layer_redundancies(
        self,
        nodes: Set[str],
        edges: Dict[Tuple[str, str], Dict[str, Set[int]]],
    ) -> Dict[Tuple[str, str], Dict[str, Set[int]]]:
        if not edges:
            return edges
        level_map = self._level_map(nodes)
        adjacency: Dict[str, Set[str]] = {}
        for src, dst in edges:
            adjacency.setdefault(src, set()).add(dst)

        removable: Set[Tuple[str, str]] = set()
        for src, dst in edges:
            src_level = level_map.get(src)
            dst_level = level_map.get(dst)
            if src_level is None or dst_level is None:
                continue
            delta = dst_level - src_level
            if abs(delta) <= 1:
                continue  # same layer or adjacent layers must stay
            direction = 1 if delta > 0 else -1
            if self._has_step_path(src, dst, direction, level_map, adjacency, exclude=(src, dst)):
                removable.add((src, dst))

        if not removable:
            return edges
        return {edge: vals for edge, vals in edges.items() if edge not in removable}

    def traverse(self,
                 root: int,
                 relation_type: Optional[str],
                 index: Optional[Union[int, Tuple[Optional[int], Optional[int]]]]) -> Tuple[Set[str], Dict[Tuple[str, str], Dict[str, Set[int]]]]:
        root = str(root)
        relation_type = relation_type.lower()
        if relation_type not in {"supergroup", "subgroup"}:
            raise ValueError(f"Unsupported relation type: {relation_type}")
        if index is None:
            index_bounds = None
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("index must be a 2-tuple: (credible_index, visible_index)")
            index_bounds = index
        else:
            index_bounds = (index, index)
        all_nodes: Set[str] = {root}
        all_edges: Dict[Tuple[str, str], Dict[str, Set[int]]] = {}
        directions = (["supergroup", "subgroup"]
                      if relation_type is None
                      else [relation_type])
        for direction in directions:
            nodes, edges = self._traverse_direction(root, direction, index_bounds)
            all_nodes.update(nodes)
            for edge, values in edges.items():
                target = all_edges.setdefault(edge, {"credible": set(), "visible": set()})
                target["credible"].update(values.get("credible", set()))
                target["visible"].update(values.get("visible", set()))
        filtered_edges = self._remove_cross_layer_redundancies(all_nodes, all_edges)
        return all_nodes, filtered_edges


@dataclass
class GraphVisualizer:
    relations: GroupRelations
    dpi: int = 300
    figsize: Tuple[float, float] = (12, 10)
    node_size: int = 1600
    font_size: int = 9
    horizontal_spacing: float = 10.0
    min_spacing: float = 2.5
    max_spacing: float = 9.0
    spacing_growth: float = 0.4
    width_scale: float = 2.0
    height_scale: float = 3.0

    def _layout(self, nodes: Set[str], spacing_scale: float = 1.0) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
        order_rows: Dict[int, List[str]] = {}
        for node in nodes:
            order_rows.setdefault(self.relations.get_order(node), []).append(node)
        sorted_orders = sorted(order_rows.keys(), reverse=True)
        max_row_size = max((len(row) for row in order_rows.values()), default=1)
        base_spacing = self.horizontal_spacing + self.spacing_growth * max(0, max_row_size - 4)
        scaled_spacing = base_spacing * spacing_scale
        global_spacing = max(self.min_spacing, min(self.max_spacing, scaled_spacing))
        layout: Dict[str, Tuple[float, float]] = {}
        for row_idx, order in enumerate(sorted_orders):
            row = sorted(order_rows[order], key=lambda g: int(g))
            if len(row) > 1:
                ratio = len(row) / max_row_size
                spacing = max(
                    self.min_spacing,
                    min(self.max_spacing, global_spacing * ratio),
                )
                width = (len(row) - 1) * spacing
                start_x = -0.5 * width
            else:
                spacing = 0.0
                start_x = 0.0
            y = len(sorted_orders) - 1 - row_idx
            for col_idx, node in enumerate(row):
                x = start_x + col_idx * spacing if len(row) > 1 else 0.0
                layout[node] = (x, y)
        return layout, max_row_size, len(sorted_orders)

    def _compute_figsize(self, max_row_size: int, layer_count: int, scale: float = 1.0) -> Tuple[float, float]:
        base_w, base_h = self.figsize
        width = max(base_w, self.width_scale * max(1, max_row_size))
        height = max(base_h, self.height_scale * max(1, layer_count))
        return width * scale, height * scale

    def _adaptive_params(self, node_count: int, layer_count: int) -> Dict[str, float]:
        node_count = max(node_count, 1)
        layer_count = max(layer_count, 1)
        node_scale = max(0.35, min(1.0, (22 / max(node_count, 6)) ** 0.5))
        layer_scale = max(0.55, min(1.0, 9 / max(layer_count, 2)))
        spacing_scale = max(0.75, min(1.25, 0.85 + 0.35 * (1 - node_scale) + 0.15 * (1 - layer_scale)))
        fig_scale = max(0.85, min(1.45, 0.85 + 0.6 * (1 - node_scale) + 0.2 * (1 - layer_scale)))
        node_size = self.node_size * node_scale
        font_size = max(6, int(self.font_size * min(node_scale * 1.15, layer_scale + 0.2)))
        edge_width = 0.8 + 0.8 * node_scale
        return {
            "node_size": node_size,
            "font_size": font_size,
            "edge_label_font_size": max(6, int(font_size * 0.9)),
            "order_font_size": max(6, int(font_size * 0.9)),
            "arrow_size": max(8, int(12 * (0.7 + 0.6 * node_scale))),
            "credible_edge_width": edge_width + 0.5,
            "edge_width": edge_width,
            "spacing_scale": spacing_scale,
            "fig_scale": fig_scale,
        }

    def draw(self,
             root: int,
             relation_type: Optional[str],
             index: Optional[Union[int, Tuple[Optional[int], Optional[int]]]],
             out_path: Optional[Path],
             show: bool,
             show_index: bool = True,
             metadata: Optional[Dict[str, Any]] = None,
             node_filter: bool = True) -> None:
        nodes, edges = self.relations.traverse(root, relation_type, index)
        if not nodes:
            raise ValueError("No nodes available to draw.")

        metadata = metadata or {}
        if node_filter and metadata:
            nodes_with_data = set(metadata.keys())
            nodes = {n for n in nodes if n in nodes_with_data}
            edges = {
                edge: vals
                for edge, vals in edges.items()
                if edge[0] in nodes and edge[1] in nodes
            }
            if not nodes:
                raise ValueError("Node filtering removed all nodes; provide metadata that includes the root.")
        layer_count_est = len({self.relations.get_order(n) for n in nodes}) or 1
        adaptive = self._adaptive_params(len(nodes), layer_count_est)
        layout, max_row_size, layer_count = self._layout(nodes, spacing_scale=adaptive["spacing_scale"])
        fig_w, fig_h = self._compute_figsize(max_row_size, layer_count, scale=adaptive["fig_scale"])

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        visible_edges = [edge for edge, data in edges.items() if data["visible"]]
        credible_edges = [edge for edge in visible_edges if edges[edge]["credible"]]
        non_credible_edges = [edge for edge in visible_edges if edge not in credible_edges]
        graph.add_edges_from(visible_edges)

        metadata = metadata or {}

        def _label_for(node: str) -> str:
            sym = self.relations.get_symbol(node)
            entries = metadata.get(node, [])

            lables = [f"#{node}-({sym})"]

            # show up to 2 entries to avoid long labels
            for entry in entries[:2]:
                mp_id = entry.get("material_id", "N/A")
                # energy_above_hull = entry.get("energy_above_hull", "N/A")
                lables.append(f"{mp_id}")

            total = len(entries)
            if total > 2:
                lables.append(f"...(total={total})")

            return "\n".join(lables)

        node_labels = {n: _label_for(n) for n in nodes}

        def _node_color(node: str) -> str:
            entries = metadata.get(node, [])
            if not entries:
                return "#f0f0dc"
            if any(entry.get("experimentally_verified") for entry in entries):
                return "#32ff65"
            return "#f0c814"

        node_colors = [_node_color(n) for n in nodes]

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)

        # fig.patch.set_alpha(0)
        # fig.patch.set_facecolor("none")
        # ax.set_facecolor("none")

        ax = plt.gca()
        ax.set_axis_off()

        nx.draw_networkx_nodes(
            graph,
            layout,
            node_color=node_colors,
            edgecolors="#555",
            linewidths=1.0,
            node_size=adaptive["node_size"],
        )

        if credible_edges:
            nx.draw_networkx_edges(
                graph,
                layout,
                edgelist=credible_edges,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=adaptive["arrow_size"],
                width=adaptive["credible_edge_width"],
                connectionstyle="arc3",
                edge_color="#2ca02c",
            )
        if non_credible_edges:
            nx.draw_networkx_edges(
                graph,
                layout,
                edgelist=non_credible_edges,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=adaptive["arrow_size"],
                width=adaptive["edge_width"],
                connectionstyle="arc3",
                edge_color="#de6666",
            )

        nx.draw_networkx_labels(graph, layout, labels=node_labels, font_size=adaptive["font_size"])

        if show_index:
            edge_labels = {
                edge: str(min(data["visible"]))
                for edge, data in edges.items()
                if data["visible"]
            }
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    graph,
                    layout,
                    edge_labels=edge_labels,
                    font_size=adaptive["edge_label_font_size"],
                    rotate=False,
                )

        order_by_y: Dict[float, int] = {}
        for node, (_, y) in layout.items():
            order_by_y.setdefault(y, self.relations.get_order(node))
        min_x = min(x for x, _ in layout.values()) if layout else 0.0
        for y, order in order_by_y.items():
            ax.text(
                min_x - 8,
                y,
                f"order {order}",
                fontsize=adaptive["order_font_size"],
                ha="left",
                va="center",
            )

        xs = [pos[0] for pos in layout.values()]
        ys = [pos[1] for pos in layout.values()]
        pad_x = max(self.min_spacing, self.horizontal_spacing)
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)

        if out_path:
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight",
                        transparent=True, pad_inches=0.05)
        if show or not out_path:
            plt.show()
        plt.close(fig)


def main(
        formula: str,
        *,
        api_key: Optional[str] = None,
        relation_type: Optional[str] = "subgroup",
        index: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        show: bool = True,
        show_index: bool = True,
        node_filter: bool = True,
) -> Optional[Path]:
    resolved_key = api_key or os.environ.get("MP_API_KEY")
    entries = load_entries(formula, api_key=resolved_key)
    metadata = create_metadata(entries)

    relation_mode = relation_type.lower() if relation_type else None
    if relation_mode not in {"supergroup", "subgroup"}:
        raise ValueError("relation_type must be 'supergroup', or 'subgroup'.")

    if relation_type == "subgroup":
        root = max(set(metadata.keys()), key=lambda g: int(g))
    else:
        root = min(set(metadata.keys()), key=lambda g: int(g))

    relations = GroupRelations.from_files()
    visualizer = GraphVisualizer(relations)

    out_path: Optional[Path] = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        relation_tag = (relation_type or "full").lower()
        out_path = output_dir / f"{normalize_formula(formula)}-{relation_tag}-tree.png"

    visualizer.draw(
        root=int(root),
        relation_type=relation_mode,
        index=index,
        out_path=out_path,
        show=show,
        show_index=show_index,
        metadata=metadata,
        node_filter=node_filter,
    )


if __name__ == "__main__":
    API_KEY = "YOUR_MATERIALS_PROJECT_API_KEY"
    formula = "CaTiO3"
    main(
        formula,
        api_key=API_KEY,
        relation_type="subgroup",
        index=(2, None),            # credible index, visible index
        output_dir=None,
        show=True
    )
