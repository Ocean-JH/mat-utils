# Mat-utils
📄 **Mat-utils** is a growing collection of small, practical Python scripts designed to help with tasks in computational materials science and AI4S (AI for Science). This repository currently includes:

---
## Overview of Tools

### 1. Ternary Element Substitution Tool
The script collection automates the process of substituting elements in ternary systems, particularly useful for exploring new materials. It includes scripts to fetch prototype structures, perform element substitutions, generate permutations, and prepare calculation folders.
- `01_Prototype_MP_API.py`: Fetches prototype structure data via the Materials Project API.
- `02_element_substitution_GeSbTe.py`: Performs element substitution in the GeSbTe system.
- `03_element_permutation.py`: Generates permutations of elements within structures.
- `04_prep_calc_folder.py`: Prepares calculation folders and input files in batch.
- `05_data_postprocessing.py`: Post-processes and organizes calculation results.
- `Element Substitution Pipeline.ipynb`: Jupyter notebook integrating the element substitution workflow.

### 2. First-principles Computational Tensile Test (FPCTT) Model Generator
The script collection automates the setup of first-principles tensile tests, generating VASP input directories for various tensile distances based on an initial structure.
> For more details on the methodology, see [Bond mobility mechanism in grain boundary embrittlement: First-principles tensile tests of Fe with a P-segregated grain boundary | Phys. Rev. B](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.82.094108).
- `abinitio_tensile_test.py`: Main script for first-principles tensile test simulations.
- `POSCAR`: Initial structure file for calculations.
- `calc_folder/Distance_*/POSCAR`: Structure files for various tensile distances.
- `calc_folder/ini_structure/POSCAR`: Backup of the initial structure.

### 3. [USPEX](https://uspex-team.org/en) Helper
The script collection is designed to analyze and visualize results from USPEX crystal structure prediction calculations.
- `uspex_convex_hull_analysis.py`: Performs convex hull analysis on USPEX results.
- `uspex_poscar_splitter.py`: Splits USPEX output structures into individual POSCAR files.

### 4. [VASP](https://www.vasp.at/) Helper
The script collection assists with VASP (Vienna Ab initio Simulation Package) calculations, including analysis and preparation of input files.
- `vasp_analyzer.py`: Analyzes VASP calculation results.
- `vasp_preparer.py`: Prepares input files for VASP calculations.

### 5. Structure Analyzer
The script collection provides a comprehensive analysis of specific crystal structures, especially on symmetry and Wyckoff sites.
- `site_env_analyzer.py`: Analyzes the local environment of sites in a crystal structure, providing insights into coordination and symmetry.

### 6. Periodic Table Mapper
The script collection provides a mapping of element-wise data to their respective positions in the periodic table, enabling intuitive visualization and analysis of element-specific properties.
- `periodic_table_mapper.py`: Visualizes element-wise data on a periodic table layout, supporting raw or normalized values from dictionaries, DataFrames, or Series.

### 7. Symmetry Statistics
The script collection analyzes the symmetry operations of crystal structures, helping to identify and classify unique symmetry elements in point and space groups.
- `point_group_explorer.py`: Lists and analyzes symmetry operations for a specified point group.
- `symmetry_elements_explorer.py`: Identifies unique symmetry elements in point groups.
- `symmetry_operations_explorer.py`: Identifies unique symmetry operations (rotation and translation) from point/space group data.
- `site_symmetry_explorer.py`: Collects statistics on site symmetries of all Wyckoff positions.

### 8. Symmetry Representations
The script collection generates matrix representations of symmetry operations.
- `symmetry_operations_interpreter.py`: Interprets the geometric meaning of symmetry operations (rotation, mirror, screw, glide, etc.) and provides details about their symmetry elements (axes or planes).
- `symop_converter.py`: Parses symmetry operation strings and converts between human-readable notation and affine matrix representations.
- `generators.py`: Generates matrix representations for symmetry operation generators of space groups.

**Data files** required (`symmetry_rep/data/`)
- `char_to_vec.json`: Mapping from character notation to vector representation for symmetry operations.
- `strings.json`: String representations of symmetry operation generators for space groups 1 to 230.

### 9. ASU Visualizer

ASU Visualizer is an interactive tool to visualize Wyckoff positions and the asymmetric unit (ASU) for a given space group. It renders unit-cell edges, 0D/1D/2D Wyckoff features and the ASU using Plotly.
- `wyckoffs_plotter.py`: Main script to visualize Wyckoff positions and ASU for a specified space group.

**Data files** required (`asu_visualizer/data/`)
- `space_group_basis.npz` — precomputed lattice bases for 1..230
- `wyckoff_map.json` — wyckoff site definitions used by the visualizer
- `std_spg_HM_map.json` — mapping from space group number to Hall number

> Acknowledgement: This module builds upon the original implementation by @rees-c, shared with permission.

---
More tools will be added over time to streamline materials design, automate repetitive tasks, and support AI-driven research.
