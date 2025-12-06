#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-12-06
Description: 
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import matplotlib.pyplot as plt
import networkx as nx


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_ORDER_PATH = DATA_ROOT / "data/sg_to_order.json"
DEFAULT_REL_PATH = DATA_ROOT / "data/super_group.json"


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class GroupRelations:
    pg_orders: Dict[str, int] = field(default_factory=dict)
    relations: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_files(cls, order_path: Path = DEFAULT_ORDER_PATH, relation_path: Path = DEFAULT_REL_PATH) -> "GroupRelations":
        return cls(load_json(order_path), load_json(relation_path))

    def get_symbol(self, group: str) -> str:
        return str(self.relations.get(str(group), {}).get("HMsymbol", "?"))

    def get_order(self, group: str) -> int:
        return int(self.pg_orders.get(str(group), 0))

    def _immediate_supergroups(self, group: str, index_filter: int) -> Dict[str, List[int]]:
        entry = self.relations.get(str(group), {})
        raw = entry.get("supergroups", {})
        if index_filter == 0:
            return {str(k): list(v) for k, v in raw.items()}
        return {str(k): list(v) for k, v in raw.items() if index_filter in v}

    def _immediate_subgroups(self, group: str, index_filter: int) -> Dict[str, List[int]]:
        subs: Dict[str, List[int]] = {}
        for candidate, info in self.relations.items():
            sg = info.get("supergroups", {})
            if str(group) in sg:
                indices = list(sg[str(group)])
                if index_filter == 0 or index_filter in indices:
                    subs[str(candidate)] = indices
        return subs

    def traverse(self, root: str, relation_type: str, index_filter: int
                 ) -> Tuple[Set[str], List[Tuple[str, str]]]:
        relation_type = relation_type.lower()
        if relation_type not in {"supergroup", "subgroup", "none"}:
            raise ValueError(f"Unsupported relation type: {relation_type}")
        root = str(root)
        nodes: Set[str] = {root}
        edges: List[Tuple[str, str]] = []
        if relation_type == "none":
            return nodes, edges
        frontier = [root]
        seen = {root}
        while frontier:
            nxt: List[str] = []
            for curr in frontier:
                if relation_type == "supergroup":
                    neighbors = self._immediate_supergroups(curr, index_filter)
                else:
                    neighbors = self._immediate_subgroups(curr, index_filter)
                for nb in neighbors.keys():
                    edge = (curr, nb) if relation_type == "supergroup" else (curr, nb)
                    if edge not in edges:
                        edges.append(edge)
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        return seen, edges


@dataclass
class GraphVisualizer:
    relations: GroupRelations
    dpi: int = 400
    node_size: int = 1300
    font_size: int = 8
    figsize: Tuple[float, float] = (6, 9)

    def _layout(self, nodes: Set[str]) -> Dict[str, Tuple[float, float]]:
        order_groups: Dict[int, List[str]] = {}
        for node in nodes:
            order_groups.setdefault(self.relations.get_order(node), []).append(node)
        sorted_orders = sorted(order_groups.keys(), reverse=True)
        layout: Dict[str, Tuple[float, float]] = {}
        for row_idx, order in enumerate(sorted_orders):
            row = sorted(order_groups[order], key=lambda g: int(g))
            spacing = 2.2
            width = (len(row) - 1) * spacing if row else 0
            start_x = -0.5 * width
            y = len(sorted_orders) - 1 - row_idx
            for col_idx, node in enumerate(row):
                layout[node] = (start_x + col_idx * spacing, y)
        return layout

    def draw(self, root: str, relation_type: str, index_filter: int,
             out_path: Optional[Path], show: bool) -> None:
        nodes, edges = self.relations.traverse(root, relation_type, index_filter)
        if not nodes:
            raise ValueError("No nodes to visualize.")
        layout = self._layout(nodes)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        labels = {n: f"{n}\n{self.relations.get_symbol(n)}" for n in nodes}

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = plt.gca()
        ax.set_axis_off()

        nx.draw_networkx_nodes(graph, layout, node_color="#f8d66d", edgecolors="black",
                               node_size=self.node_size, linewidths=0.8)
        nx.draw_networkx_edges(graph, layout, arrows=True, arrowstyle="-|>", arrowsize=14,
                               width=1.2, connectionstyle="arc3")
        nx.draw_networkx_labels(graph, layout, labels=labels, font_size=self.font_size)

        order_labels = {}
        for node, (_, y) in layout.items():
            order_labels.setdefault(y, self.relations.get_order(node))
        for y, order in order_labels.items():
            ax.text(min(x for x, _ in layout.values()) - 2.5, y, f"order {order}",
                    fontsize=self.font_size, va="center", ha="left")

        xs = [pos[0] for pos in layout.values()]
        ys = [pos[1] for pos in layout.values()]
        ax.set_xlim(min(xs) - 3.0, max(xs) + 3.0)
        ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, bbox_inches="tight", dpi=self.dpi)
        if show or not out_path:
            plt.show()
        plt.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize spatial group relations.")
    parser.add_argument("group", type=str, help="Spatial group number (e.g., 228)")
    parser.add_argument("--relation", choices=["supergroup", "subgroup"],
                        default=None, help="Relation type to traverse.")
    parser.add_argument("--index", type=int, default=0,
                        help="Index filter (0 keeps all indices).")
    parser.add_argument("--orders", type=Path, default=DEFAULT_ORDER_PATH,
                        help="Path to sg_to_order.json.")
    parser.add_argument("--relations", type=Path, default=DEFAULT_REL_PATH,
                        help="Path to super_group.json.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional output image path (PNG recommended).")
    parser.add_argument("--show", action="store_true",
                        help="Force showing the plot even when saving.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    # args = parse_args(argv)
    group_rel = GroupRelations.from_files()
    visualizer = GraphVisualizer(group_rel)
    try:
        visualizer.draw(root=str(62),
                        relation_type="subgroup",
                        index_filter=0,
                        out_path=None,
                        show=True)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
