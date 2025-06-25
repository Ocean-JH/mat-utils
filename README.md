# Mat-utils
📄 **Mat-utils** is a growing collection of small, practical Python scripts designed to help with tasks in computational materials science and AI4S (AI for Science). This repository currently includes:

---
## Overview of Tools

### 1. Ternary Element Substitution Tool
This tool automates the process of substituting elements in ternary systems, particularly useful for exploring new materials. It includes scripts to fetch prototype structures, perform element substitutions, generate permutations, and prepare calculation folders.
- `01_Prototype_MP_API.py`: Fetches prototype structure data via the Materials Project API.
- `02_element_substitution_GeSbTe.py`: Performs element substitution in the GeSbTe system.
- `03_element_permutation.py`: Generates permutations of elements within structures.
- `04_prep_calc_folder.py`: Prepares calculation folders and input files in batch.
- `05_data_postprocessing.py`: Post-processes and organizes calculation results.
- `Element Substitution Pipeline.ipynb`: Jupyter notebook integrating the element substitution workflow.

### 2. First-principles Computational Tensile Test (FPCTT) Model Generator
This tool automates the setup of first-principles tensile tests, generating VASP input directories for various tensile distances based on an initial structure.
- `abinitio_tensile_test.py`: Main script for first-principles tensile test simulations.
- `POSCAR`: Initial structure file for calculations.
- `calc_folder/Distance_*/POSCAR`: Structure files for various tensile distances.
- `calc_folder/ini_structure/POSCAR`: Backup of the initial structure.

### 3. [USPEX](https://uspex-team.org/en) Helper
This collection of scripts is designed to analyze and visualize results from USPEX crystal structure prediction calculations.
- `uspex_convex_hull_analysis.py`: Performs convex hull analysis on USPEX results.
- `uspex_poscar_splitter.py`: Splits USPEX output structures into individual POSCAR files.

### 4. VASP Helper
- `vasp_analyzer.py`: Analyzes VASP calculation results.
- `vasp_preparer.py`: Prepares input files for VASP calculations.

---
More tools will be added over time to streamline materials design, automate repetitive tasks, and support AI-driven research.

## *Reference*

[1]: [Bond mobility mechanism in grain boundary embrittlement: First-principles tensile tests of Fe with a P-segregated grain boundary | Phys. Rev. B](https://journals-aps-org.remotexs.ntu.edu.sg/prb/abstract/10.1103/PhysRevB.82.094108)
