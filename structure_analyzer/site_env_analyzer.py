#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-08-06
Description: This script collects site information from a structure.
"""
import numpy as np

from pymatgen.core.structure import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments


def _format_coordination_environments(env_list, csm_cutoff=2.5, digits=6):
    """Format coordination environments into {symbol: fraction} dict."""
    if not env_list:
        return {}
    return {
        env['ce_symbol']: round(float(env['ce_fraction']), digits)
        for env in env_list if env['csm'] < csm_cutoff
    }

def get_site_groups(structure: Structure, sga: SpacegroupAnalyzer = None):
    """Return grouping info by symmetry equivalence."""
    sga = SpacegroupAnalyzer(structure) if sga is None else sga
    sym_data = sga.get_symmetry_dataset()
    eq_pos = sym_data.equivalent_atoms

    unique_indices = np.unique(eq_pos)

    site_groups = []
    for rep_idx in unique_indices:
        indices = np.where(eq_pos == rep_idx)[0]
        rep_site = structure.sites[rep_idx]
        group = {
            "rep_idx": int(rep_idx),
            "rep_site": rep_site,
            "site_indices": indices.tolist(),
            "equi_sites": [structure.sites[i] for i in indices],
            "site_element": [sp.symbol for sp in rep_site.species.elements],
            "site_occupancy": [round(rep_site.species.get_wt_fraction(el), 6) for el in rep_site.species.elements],
            "multiplicity": len(indices),
            "wyckoff_letter": sym_data.wyckoffs[rep_idx],
            "site_symmetry": sym_data.site_symmetry_symbols[rep_idx],
        }
        site_groups.append(group)
    return site_groups

def load_env_analyzers(structure: Structure):
    """Precompute CrystalNN and ChemEnv analyzers for reuse."""
    cnn = CrystalNN()
    lgf = LocalGeometryFinder()
    # lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True)
    lgf.setup_structure(structure)

    se = lgf.compute_structure_environments()
    strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
    lse = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)

    return cnn, lse

def get_coord_envs(structure: Structure, site_index: int, cnn: CrystalNN = None, lse: LightStructureEnvironments = None):
    """Get coordination number and environments for a specific site index."""
    if cnn is None or lse is None:
        cnn, lse = load_env_analyzers(structure)

    # Get coordination number
    cn = cnn.get_cn(structure, site_index, on_disorder='take_max_species')

    # Get coordination environments
    ce = lse.coordination_environments[site_index]
    ce_formatted = _format_coordination_environments(ce)

    return cn, ce_formatted


def main(structure: Structure):
    """Return global and site-wise symmetry and environment info."""
    # Get space group information
    sga = SpacegroupAnalyzer(structure)
    site_groups = get_site_groups(structure, sga)

    # Load analyzers
    cnn, lse = load_env_analyzers(structure)

    global_info = {
        "crystal_system": sga.get_crystal_system(),
        "space_group_number": sga.get_space_group_number(),
        "space_group_symbol": sga.get_space_group_symbol(),
        "site_info": [],
    }

    for group in site_groups:
        idx = group["rep_idx"]
        # Coordination analysis
        cn, coord_envs = get_coord_envs(structure, idx, cnn, lse)

        site_info = {
            "element": group['site_element'],
            'occupancy': group['site_occupancy'],
            "multiplicity": group['multiplicity'],
            "wyckoff_letter": group['wyckoff_letter'],
            "site_symmetry": group["site_symmetry"],
            "coordination_number": cn,
            "coordination_environments": coord_envs,
        }

        global_info["site_info"].append(site_info)

    return global_info


if __name__ == "__main__":
    # Example usage
    structure = Structure.from_file("path/to/your/structure_file.cif")
    info = main(structure)
    print(info)