#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
            Step: 01
            Data mining of Ge-Sb-Te system with valency constraint.
            Author: Jianghai@BUAA
"""

import os
import pandas as pd

from math import gcd
from mp_api.client import MPRester

API_KEY = "type_your_API_key_here"  # Replace with your Materials Project API key

"""
            Periodic Table of Elements
"""
# Elements = ['H', 'He',
#             'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
#             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
#             'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
#             'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
#             'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
#             'Fr', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', ' Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
#             'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
#             'Ac', 'Th', 'Pa', 'U', ' Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']


"""
            Stoichiometry satisfied the valency constraint
"""


def coprime(a, b, c):
    return gcd(a, b, c) == 1


Stoichiometry = []
for x in range(1, 20):
    for y in range(1, 20):
        for z in range(1, 20):
            if 2 * x + 3 * y - 2 * z == 0:
                if coprime(x, y, z):
                    Stoichiometry.append([x, y, z])

"""
            Data retrieval
"""
for formula in Stoichiometry:
    with MPRester(API_KEY) as mpr:
        # list_of_available_fields = mpr.materials.summary.available_fields
        # print(list_of_available_fields)
        """
        ['builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty',
        'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'property_name',
        'material_id', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'structure',
        'task_ids', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull',
        'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'xas', 'grain_boundaries', 'band_gap',
        'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'es_source_calc_id', 'bandstructure', 'dos',
        'dos_energy_up', 'dos_energy_down', 'is_magnetic', 'ordering', 'total_magnetization',
        'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites',
        'num_unique_magnetic_sites', 'types_of_magnetic_species', 'k_voigt', 'k_reuss', 'k_vrh', 'g_voigt', 'g_reuss',
        'g_vrh', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max',
        'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy', 'weighted_work_function',
        'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 'has_props', 'theoretical',
        'database_IDs']
        """

        docs = mpr.materials.summary.search(formula=["*{}*{}*{}".format(formula[0], formula[1], formula[2])],
                                            fields=['material_id', 'database_IDs', 'nelements', 'chemsys',
                                                    'formula_anonymous', 'formula_pretty', 'composition',
                                                    'symmetry', 'energy_per_atom', 'formation_energy_per_atom',
                                                    'energy_above_hull', 'is_stable', 'theoretical', 'volume',
                                                    'density', 'possible_species', 'decomposes_to',
                                                    'deprecated', 'origins', 'warnings', 'last_updated',
                                                    'structure'])

        MP_Prototype_Info = [
            [doc.material_id, doc.database_IDs, doc.nelements, doc.chemsys, doc.formula_anonymous,
             doc.formula_pretty, doc.composition, doc.symmetry, doc.energy_per_atom,
             doc.formation_energy_per_atom, doc.energy_above_hull, doc.is_stable, doc.theoretical, doc.volume,
             doc.density, doc.possible_species, doc.decomposes_to, doc.deprecated, doc.origins, doc.warnings,
             doc.last_updated] for doc in docs]

    MP_Prototype_Info = pd.DataFrame(MP_Prototype_Info,
                                     columns=['material_id', 'database_IDs', 'nelements', 'chemsys',
                                              'formula_anonymous', 'formula_pretty', 'composition', 'symmetry',
                                              'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull',
                                              'is_stable', 'theoretical', 'volume', 'density', 'possible_species',
                                              'decomposes_to', 'deprecated', 'origins', 'warnings', 'last_updated'])

    if not os.path.exists(r'~\MP_Prototype_Info.csv'):
        MP_Prototype_Info.to_csv(r'~\MP_Prototype_Info.csv',
                                 mode='a', index=False)
    else:
        MP_Prototype_Info.to_csv(r'~\MP_Prototype_Info.csv',
                                 mode='a', index=False, header=False)

info = pd.read_csv(r'~\MP_Prototype_Info.csv')
info_symmetry = info['symmetry'].str.split('\'', expand=True)[[1, 3, 4, 5]]
info_spg = info_symmetry[4].str.split('=', expand=True)[1].str.split(expand=True)[0]
info_sym = pd.concat([info_symmetry, info_spg], axis=1)
info = pd.concat([info, info_sym], axis=1)
info = info.rename(columns={0: 'space_group_number', 1: 'crystal_system', 3: 'space_group', 5: 'point_group'})
info.drop([4, 'symmetry'], axis=1, inplace=True)
info.to_csv(r'~\MP_Prototype_Info.csv', mode='w', index=False)
