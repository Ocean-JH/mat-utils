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
from typing import Any, Dict, List, Set, Tuple, Optional

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
        query_entries(formula, api_key=api_key, save_structures=save_structures)

    return entries

def create_metadata(entries: List[Dict]) -> Dict[str, Any]:
    """Build metadata dictionary from entries."""
    tree_metadata: Dict[str, List[Dict[str, Any]]] = {}
    space_groups: Set[str] = set()

    for entry in entries:
        sg = entry["spacegroup"].get("number")

        if sg is None:
            continue

        sg_key = str(int(sg))
        space_groups.add(sg_key)

        meta = {
            "material_id": entry.get("material_id"),
            "formula": entry.get("formula"),
            "formation_energy_per_atom": entry.get("formation_energy_per_atom"),
            "energy_above_hull": entry.get("energy_above_hull"),
            "is_stable": entry.get("is_stable"),
            "experimentally_verified": entry.get("experimentally_verified"),
            "database_IDs": entry.get("database_IDs")
        }

        tree_metadata.setdefault(sg_key, []).append(meta)

    for entries in tree_metadata.values():
        entries.sort(
            key=lambda m: (
                m["energy_above_hull"] is None,
                m["energy_above_hull"] if m["energy_above_hull"] is not None else float("inf")
            )
        )

    return tree_metadata


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
                   ) -> List[Tuple[str, List[int]], List[int]]:
        group = str(group)
        relation_type = relation_type.lower()
        if relation_type == "supergroup":
            raw = self.super_rel.get(group, {}).get("supergroups", {})
        elif relation_type == "subgroup":
            raw = self.sub_rel.get(group, {})
        else:
            raise ValueError(f"Relation type must be 'supergroup' or 'subgroup', got: {relation_type}")

        credible_index, visible_index = (index_bounds if index_bounds is not None else (2, None))

        results: List[Tuple[str, List[int]], List[int]] = []
        for neighbor, indices in raw.items():
            if neighbor == group:
                continue
            cred_idx = [idx for idx in indices
                        if (credible_index is None or idx <= credible_index)]
            vis_idx = [idx for idx in indices
                       if (visible_index is None or idx <= visible_index)]
            if not cred_idx and not vis_idx:
                continue
            results.append((str(neighbor), cred_idx, vis_idx))

        return results

    def _traverse_direction(self,
                            root: str,
                            relation_type: str,
                            index_bounds: Optional[Tuple[Optional[int], Optional[int]]] = None
                            ) -> Tuple[Set[str], Dict[Tuple[str, str], Set[int]]]:
        visited: Set[str] = {root}
        edges: Dict[Tuple[str, str], Set[int]] = {}
        queue: deque[str] = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor, indices in self._neighbors(current, relation_type, index_bounds):
                edge = (current, neighbor)
                edges.setdefault(edge, set()).update(indices)
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
        edges: Dict[Tuple[str, str], Set[int]],
    ) -> Dict[Tuple[str, str], Set[int]]:
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
                 index: Optional[int]) -> Tuple[Set[str], Dict[Tuple[str, str], Set[int]]]:
        root = str(root)
        relation_type = relation_type.lower()
        if relation_type not in {"supergroup", "subgroup"}:
            raise ValueError(f"Unsupported relation type: {relation_type}")
        all_nodes: Set[str] = {root}
        all_edges: Dict[Tuple[str, str], Set[int]] = {}
        directions = (["supergroup", "subgroup"]
                      if relation_type is None
                      else [relation_type])
        for direction in directions:
            nodes, edges = self._traverse_direction(root, direction, index)
            all_nodes.update(nodes)
            for edge, values in edges.items():
                all_edges.setdefault(edge, set()).update(values)
        filtered_edges = self._remove_cross_layer_redundancies(all_nodes, all_edges)
        return all_nodes, filtered_edges


@dataclass
class GraphVisualizer:
    relations: GroupRelations
    dpi: int = 300
    figsize: Tuple[float, float] = (12, 10)
    node_size: int = 1200
    font_size: int = 9
    horizontal_spacing: float = 10.0
    min_spacing: float = 2.5
    max_spacing: float = 9.0
    spacing_growth: float = 0.4
    width_scale: float = 2.0
    height_scale: float = 3.0

    def _layout(self, nodes: Set[str]) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
        order_rows: Dict[int, List[str]] = {}
        for node in nodes:
            order_rows.setdefault(self.relations.get_order(node), []).append(node)
        sorted_orders = sorted(order_rows.keys(), reverse=True)
        max_row_size = max((len(row) for row in order_rows.values()), default=1)
        global_spacing = self.horizontal_spacing + self.spacing_growth * max(0, max_row_size - 4)
        global_spacing = max(self.min_spacing, min(self.max_spacing, global_spacing))
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

    def _compute_figsize(self, max_row_size: int, layer_count: int) -> Tuple[float, float]:
        base_w, base_h = self.figsize
        width = max(base_w, self.width_scale * max(1, max_row_size))
        height = max(base_h, self.height_scale * max(1, layer_count))
        return width, height

    def draw(self,
             root: int,
             relation_type: Optional[str],
             index: Optional[int],
             out_path: Optional[Path],
             show: bool,
             show_edge_labels: bool = True) -> None:
        nodes, edges = self.relations.traverse(root, relation_type, index)
        if not nodes:
            raise ValueError("No nodes available to draw.")
        layout, max_row_size, layer_count = self._layout(nodes)
        fig_w, fig_h = self._compute_figsize(max_row_size, layer_count)

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges.keys())

        labels = {n: f"{n}\n{self.relations.get_symbol(n)}" for n in nodes}

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi)
        ax = plt.gca()
        ax.set_axis_off()

        nx.draw_networkx_nodes(
            graph,
            layout,
            node_color="#fee9b2",
            edgecolors="#555",
            linewidths=0.8,
            node_size=self.node_size,
        )
        nx.draw_networkx_edges(
            graph,
            layout,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            width=1.2,
            connectionstyle="arc3",
            edge_color="#666",
        )
        nx.draw_networkx_labels(graph, layout, labels=labels, font_size=self.font_size)

        if show_edge_labels and edges:
            edge_labels = {
                edge: str(min(values))
                for edge, values in edges.items()
            }
            nx.draw_networkx_edge_labels(
                graph,
                layout,
                edge_labels=edge_labels,
                font_size=max(self.font_size - 1, 6),
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
                fontsize=self.font_size,
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


if __name__ == "__main__":
    API_KEY = "TY4CfoGAcNxKcd0Di0AiErwrrLZVVhsz"

    load_entries("CaTiO3", API_KEY)
    # raise SystemExit(main())
    # relations = GroupRelations.from_files()
    # visualizer = GraphVisualizer(relations)
    # visualizer.draw(
    #     root=62,
    #     relation_type="subgroup",
    #     index=None,
    #     out_path=None,
    #     show=True,
    #     show_edge_labels=True,
    # )
