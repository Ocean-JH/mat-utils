#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-07-12
Description: Collect statistics on site symmetries of all Wyckoff positions for the 230 space groups using `pyxtal`.
"""
from collections import defaultdict, Counter
from typing import Dict, Set, Tuple, Optional, List

from pyxtal.symmetry import Group


def find_unique_site_symmetries() -> Set[str]:
    """
    Find all unique site symmetries from the 230 space groups.

    Returns
    -------
    unique_syms : set[str]
        A set of unique site symmetries (normalised).
    """
    unique_ss: Set[str] = set()
    for sg_number in range(1, 231):
        group = Group(sg_number)
        for wp in group.Wyckoff_positions:
            site_sym = wp.get_site_symmetry_object().name
            unique_ss.add(site_sym)

    return unique_ss


def site_symmetry_stats() -> Tuple[Dict[str, Dict[int, Dict[str, Set[str]]]], Counter]:
    """
        Scan the 230 space groups and gather site‑symmetry statistics.

        Returns
        -------
        unique_syms : set[str]
            All distinct (normalised) site symmetries.
        sym_details : dict
            site_sym -> {sg_number ->
                            {"symbol": space‑group symbol,
                             "wyckoff_letters": set[str]}}
        sym_counter : Counter
            Counts of how many Wyckoff positions carry each site symmetry.
    """
    sym_counter: Counter = Counter()
    # site_sym → {sg_number → {"symbol": str, "wyckoff_letters": set}}
    sym_details: Dict[str, Dict[int, Dict[str, Set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"symbol": "", "wyckoff_letters": set()})
    )

    for sg_number in range(1, 231):
        group = Group(sg_number)
        for wp in group.Wyckoff_positions:  # walk through Wyckoff positions
            site_sym = wp.get_site_symmetry_object().name
            sym_counter[site_sym] += 1

            info = sym_details[site_sym][sg_number]
            info["symbol"] = group.symbol  # e.g. 'Pnma'
            info["wyckoff_letters"].add(wp.letter)  # accumulate letters (a, b, ...)

    return sym_details, sym_counter


def get_stats(unique_ss: Set[str],
              ss_details: Dict[str, Dict[int, Dict[str, Set[str]]]],
              ss_counter: Counter,
              *,
              target_sym: Optional[str] = None,
              top_n: Optional[int] = None) -> None:
    """
    Display statistics in a human‑readable table.

    Parameters
    ----------
    target_sym : str | None
        Print only this site symmetry (case‑sensitive, e.g. 'mmm' or 'm-3m').
    top_n : int | None
        If `target_sym` is None, print only the first *n* most frequent symmetries.
    """
    if target_sym:
        if target_sym not in ss_details:
            raise ValueError(f"Site symmetry '{target_sym}' not found in the data.")
        sym_list: List[str] = [target_sym]
    else:
        sym_list = sorted(unique_ss, key=ss_counter.get, reverse=True)
        if top_n:
            sym_list = sym_list[:top_n]

    if not target_sym:
        print(f"Total unique site symmetries: {len(unique_ss)}\n")

    for sym in sym_list:
        print(f"Site symmetry '{sym}' — {ss_counter[sym]} Wyckoff positions")
        for sg_num in sorted(ss_details[sym]):
            info = ss_details[sym][sg_num]
            letters = ", ".join(sorted(info["wyckoff_letters"]))
            print(f"  • SG {sg_num:3d} ({info['symbol']:>5s}): {letters}")
        print("-" * 60)


if __name__ == "__main__":
    unique_ss = find_unique_site_symmetries()
    details, counter = site_symmetry_stats()
    get_stats(unique_ss, details, counter, target_sym=None, top_n=None)
