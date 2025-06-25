#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-06-25
Description: Prepare VASP RELAX input files from CIF files.
"""
import os
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet


def generate_relax_input(cif_path: str, output_root: str) -> None:
    """
    Generate VASP input files from a CIF file using MPRelaxSet.

    Parameters:
        cif_path (str): Path to the .cif file.
        output_root (str): Root directory where the VASP input files will be saved.
    """
    structure = Structure.from_file(cif_path)

    vasp_input_set = MPRelaxSet(structure)

    struct_name = os.path.splitext(os.path.basename(cif_path))[0]
    struct_folder = os.path.join(output_root, struct_name)
    os.makedirs(struct_folder, exist_ok=True)

    vasp_input_set.write_input(struct_folder)

def batch_generate_vasp_inputs(cif_folder: str, output_root: str) -> None:
    """
    Batch generate VASP input files from all CIF files in a folder.

    Parameters:
        cif_folder (str): Folder containing .cif files.
        output_root (str): Root directory where the VASP input files will be saved.
    """
    os.makedirs(output_root, exist_ok=True)

    # Recursively find all .cif files
    cif_files = list(Path(cif_folder).rglob("*.cif"))
    print(f"Found {len(cif_files)} CIF files")

    for i, cif_path in enumerate(cif_files):
        generate_relax_input(str(cif_path), output_root)


if __name__ == "__main__":
    cif_folder = "./"
    root_directory = "./"
    batch_generate_vasp_inputs(cif_folder, root_directory)
    print(f"VASP input files generated in {root_directory}")
