#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-12-06
Description:
    Visualize space-group relations (super/sub) as high-resolution vertical graphs.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_ORDER_PATH = DATA_DIR / "sg_to_order.json"
DEFAULT_SUPER_PATH = DATA_DIR / "supergroup.json"
DEFAULT_SUB_PATH = DATA_DIR / "subgroup.json"


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required data file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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
            orders=load_json(order_path),
            super_rel=load_json(super_path),
            sub_rel=load_json(sub_path),
        )

    def get_symbol(self, group: str) -> str:
        entry = self.super_rel.get(str(group))
        return entry.get("HMsymbol", "?") if entry else "?"

    def get_order(self, group: str) -> int:
        return int(self.orders.get(str(group), 0))

    def _neighbors(self, group: str, relation_type: str, index: int = None) -> List[Tuple[str, List[int]]]:
        group = str(group)
        relation_type = relation_type.lower()
        if relation_type == "supergroup":
            raw = self.super_rel.get(group, {}).get("supergroups", {})
        elif relation_type == "subgroup":
            raw = self.sub_rel.get(group, {})
        else:
            return []
        results: List[Tuple[str, List[int]]] = []
        for neighbor, indices in raw.items():
            if neighbor == group:
                continue
            filtered = [list(indices)[0]]
            results.append((str(neighbor), filtered))
        return results

    def _traverse_direction(self,
                            root: str,
                            relation_type: str,
                            index: int) -> Tuple[Set[str], Dict[Tuple[str, str], Set[int]]]:
        visited: Set[str] = {root}
        edges: Dict[Tuple[str, str], Set[int]] = {}
        queue: deque[str] = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor, indices in self._neighbors(current, relation_type, index):
                edge = (current, neighbor)
                edges.setdefault(edge, set()).update(indices)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited, edges

    def traverse(self,
                 root: str,
                 relation_type: Optional[str],
                 index: int) -> Tuple[Set[str], Dict[Tuple[str, str], Set[int]]]:
        root = str(root)
        relation_type = (relation_type or "none").lower()
        if relation_type not in {"supergroup", "subgroup", "none"}:
            raise ValueError(f"Unsupported relation type: {relation_type}")
        all_nodes: Set[str] = {root}
        all_edges: Dict[Tuple[str, str], Set[int]] = {}
        directions = (["supergroup", "subgroup"]
                      if relation_type == "none"
                      else [relation_type])
        for direction in directions:
            nodes, edges = self._traverse_direction(root, direction, index)
            all_nodes.update(nodes)
            for edge, values in edges.items():
                all_edges.setdefault(edge, set()).update(values)
        return all_nodes, all_edges


@dataclass
class GraphVisualizer:
    relations: GroupRelations
    dpi: int = 400
    figsize: Tuple[float, float] = (7.0, 11.0)
    node_size: int = 1500
    font_size: int = 9
    horizontal_spacing: float = 5.0

    def _layout(self, nodes: Set[str]) -> Dict[str, Tuple[float, float]]:
        order_rows: Dict[int, List[str]] = {}
        for node in nodes:
            order_rows.setdefault(self.relations.get_order(node), []).append(node)
        sorted_orders = sorted(order_rows.keys(), reverse=True)
        layout: Dict[str, Tuple[float, float]] = {}
        for row_idx, order in enumerate(sorted_orders):
            row = sorted(order_rows[order], key=lambda g: int(g))
            spacing = self.horizontal_spacing
            width = (len(row) - 1) * spacing if row else 0.0
            start_x = -0.5 * width
            y = len(sorted_orders) - 1 - row_idx
            for col_idx, node in enumerate(row):
                layout[node] = (start_x + col_idx * spacing, y)
        return layout

    def draw(self,
             root: str,
             relation_type: Optional[str],
             index: int,
             out_path: Optional[Path],
             show: bool,
             show_edge_labels: bool = True) -> None:
        nodes, edges = self.relations.traverse(root, relation_type, index)
        if not nodes:
            raise ValueError("No nodes available to draw.")
        layout = self._layout(nodes)

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges.keys())

        labels = {n: f"{n}\n{self.relations.get_symbol(n)}" for n in nodes}

        plt.figure(figsize=self.figsize, dpi=self.dpi)
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
                min_x - 2.5,
                y,
                f"order {order}",
                fontsize=self.font_size,
                ha="left",
                va="center",
            )

        xs = [pos[0] for pos in layout.values()]
        ys = [pos[1] for pos in layout.values()]
        ax.set_xlim(min(xs) - 3.0, max(xs) + 3.0)
        ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        if show or not out_path:
            plt.show()
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Space-group relation visualizer.")
    parser.add_argument("group", type=str, help="Space group number (e.g., 229).")
    parser.add_argument("--relation", choices=["supergroup", "subgroup", "none"],
                        default="none", help="Relation type (draw both when 'none').")
    parser.add_argument("--index", type=int, default=0,
                        help="Index filter (0 keeps all indices).")
    parser.add_argument("--orders", type=Path, default=DEFAULT_ORDER_PATH,
                        help="Path to sg_to_order.json.")
    parser.add_argument("--supergroups", type=Path, default=DEFAULT_SUPER_PATH,
                        help="Path to supergroup.json.")
    parser.add_argument("--subgroups", type=Path, default=DEFAULT_SUB_PATH,
                        help="Path to subgroup.json.")
    parser.add_argument("--out", type=Path,
                        help="Optional output path for the figure (PNG recommended).")
    parser.add_argument("--show", action="store_true",
                        help="Show the figure even when saving.")
    parser.add_argument("--no-edge-labels", action="store_false", dest="edge_labels",
                        help="Disable drawing index labels on edges.")
    parser.set_defaults(edge_labels=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    relations = GroupRelations.from_files(args.orders, args.supergroups, args.subgroups)
    visualizer = GraphVisualizer(relations)
    visualizer.draw(
        root=args.group,
        relation_type=None if args.relation == "none" else args.relation,
        index=args.index,
        out_path=args.out,
        show=args.show,
        show_edge_labels=args.edge_labels,
    )
    return 0


if __name__ == "__main__":
    # raise SystemExit(main())
    relations = GroupRelations.from_files()
    visualizer = GraphVisualizer(relations)
    visualizer.draw(
        root=62,
        relation_type="subgroup",
        index=0,
        out_path=None,
        show=True,
        show_edge_labels=True,
    )