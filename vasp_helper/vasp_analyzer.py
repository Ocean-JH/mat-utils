#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-06-24
Description: Extract VASP calculation results and save as pickle and JSON files.
"""
import os
import pickle
import json
from pymatgen.io.vasp import Vasprun
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_computed_entry(vasprun_path: Path, entry_id=None):
    """
    Get ComputedEntry object from vasprun.xml.

    Parameters:
    -----------
    vasprun_path : Path
        The path of 'vasprun.xml' containing VASP calculation results.
    entry_id : str, optional
        Optional entry ID to use for the ComputedEntry.

    Returns:
    --------
    ComputedEntry
        The computed entry with energy, composition, and some other information.
    """
    folder_name = vasprun_path.parent.name

    # Load vasprun.xml file
    vasprun = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
    entry = vasprun.get_computed_entry(entry_id=folder_name if entry_id is None else entry_id)
    return entry


def batch_extract_vasp_results(root_directory, output_entry="entry.pkl", output_results="results_summary.json"):
    """
    Batch extract structural energy, composition and space group information from VASP calculation folders
    
    Parameters:
    -----------
    root_directory : str
        Root directory containing VASP calculation folders
    output_entry : str
        Output pickle filename for complete ComputedEntry objects
    output_results : str
        Output JSON filename for summary data
    
    Returns:
    --------
    list
        List containing extraction results
    """
    
    results = []
    results_summary = []
    failed_calculations = []
    
    # Recursively find all vasprun.xml files
    vasprun_files = list(Path(root_directory).rglob("vasprun.xml"))
    
    print(f"Found {len(vasprun_files)} vasprun.xml files")
    
    for i, vasprun_path in enumerate(vasprun_files):
        calc_dir = vasprun_path.parent
        relative_path = calc_dir.relative_to(root_directory)
        
        # Use folder name as entry_id
        entry_id = calc_dir.name
        
        print(f"Processing ({i+1}/{len(vasprun_files)}): {relative_path} (entry_id: {entry_id})")
        
        try:
            # Read vasprun.xml file
            vasprun = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)

            # Get computed entry using folder name as entry_id
            computed_entry = vasprun.get_computed_entry(entry_id=entry_id)
            
            # Get final structure
            final_structure = vasprun.final_structure
            
            # Store complete computed entry for pickle
            entry_data = {
                'entry_id': entry_id,
                'computed_entry': computed_entry,
                'vasprun_data': {
                    'converged': vasprun.converged,
                    'efermi': vasprun.efermi if hasattr(vasprun, 'efermi') else None,
                }
            }
            results.append(entry_data)
            
            # Create JSON-serializable summary
            summary = {
                'entry_id': entry_id,
                'formula': computed_entry.composition.reduced_formula,
                'composition': computed_entry.composition.formula,
                "uncorrected_energy": computed_entry.uncorrected_energy,
                'uncorrected_energy_per_atom': computed_entry.uncorrected_energy / len(final_structure),
                'correction': float(computed_entry.correction) if hasattr(computed_entry, 'correction') else 0.0,
                'final_energy': float(computed_entry.energy),
                'energy_per_atom': float(computed_entry.energy_per_atom),
                'space_group_number': final_structure.get_space_group_info()[0],
                'space_group_symbol': final_structure.get_space_group_info()[1],
                'lattice_a': float(final_structure.lattice.a),
                'lattice_b': float(final_structure.lattice.b),
                'lattice_c': float(final_structure.lattice.c),
                'lattice_alpha': float(final_structure.lattice.alpha),
                'lattice_beta': float(final_structure.lattice.beta),
                'lattice_gamma': float(final_structure.lattice.gamma),
                'volume': float(final_structure.volume),
                'density': float(final_structure.density),
                'num_atoms': len(final_structure),
                'converged': vasprun.converged,
                'efermi': float(vasprun.efermi) if hasattr(vasprun, 'efermi') and vasprun.efermi is not None else None,
            }
            results_summary.append(summary)
            
        except Exception as e:
            error_info = {
                'entry_id': entry_id,
                'calculation_path': str(relative_path),
                'error': str(e)
            }
            failed_calculations.append(error_info)
            print(f"  Error: {e}")
    
    # Save complete results to pickle
    if results:
        with open(output_entry, 'wb') as f:
            pickle.dump(results, f)
        print(f"Complete ComputedEntry objects saved to: {output_entry}")
        
        # Save summary to JSON
        with open(output_results, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Summary data saved to: {output_results}")
        
        print(f"\nSuccessfully processed {len(results)} calculations")
        
        # Display statistics
        print("\n=== Statistics ===")
        unique_formulas = len(set(r['formula'] for r in results_summary))
        energies = [r['final_energy'] for r in results_summary]
        converged_count = sum(1 for r in results_summary if r['converged'])
        
        print(f"Number of unique compositions: {unique_formulas}")
        print(f"Energy range: {min(energies):.3f} to {max(energies):.3f} eV")
        print(f"Number of converged calculations: {converged_count}")
        
        # Display space group distribution
        print("\nSpace group distribution:")
        space_groups = {}
        for r in results_summary:
            sg = r['space_group_symbol']
            space_groups[sg] = space_groups.get(sg, 0) + 1
        
        # Sort by frequency and show top 10
        sorted_sg = sorted(space_groups.items(), key=lambda x: x[1], reverse=True)[:10]
        for sg, count in sorted_sg:
            print(f"  {sg}: {count}")
    
    else:
        print("No calculations processed successfully")
    
    # Report failed calculations
    if failed_calculations:
        print(f"\nFailed calculations ({len(failed_calculations)}):")
        with open("failed_calculations.json", 'w') as f:
            json.dump(failed_calculations, f, indent=2)
        
        for fail in failed_calculations[:5]:  # Show only first 5
            print(f"  {fail['calculation_path']}: {fail['error']}")
        if len(failed_calculations) > 5:
            print(f"  ... and {len(failed_calculations)-5} more failed calculations")
        print("Failed list saved to: failed_calculations.json")
    
    return results, results_summary

def load_vasp_entries(pickle_file="entry.pkl"):
    """
    Load complete ComputedEntry objects from pickle file
    
    Parameters:
    -----------
    pickle_file : str
        Pickle filename containing ComputedEntry objects
    
    Returns:
    --------
    list
        List containing complete ComputedEntry data
    """
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def filter_and_analyze_results(results_summary, energy_cutoff=None, space_groups=None):
    """
    Filter and analyze results using summary data
    
    Parameters:
    -----------
    results_summary : list
        List containing VASP summary results
    energy_cutoff : float, optional
        Energy cutoff value (eV/atom), only keep results below this value
    space_groups : list, optional
        List of specified space groups, only keep results with these space groups
    """
    
    filtered_results = results_summary.copy()
    
    # Filter by energy
    if energy_cutoff is not None:
        initial_count = len(filtered_results)
        filtered_results = [r for r in filtered_results if r['energy_per_atom'] <= energy_cutoff]
        print(f"Energy filtering (<= {energy_cutoff} eV/atom): {initial_count} -> {len(filtered_results)}")
    
    # Filter by space group
    if space_groups is not None:
        initial_count = len(filtered_results)
        filtered_results = [r for r in filtered_results if r['space_group_symbol'] in space_groups]
        print(f"Space group filtering ({space_groups}): {initial_count} -> {len(filtered_results)}")
    
    # Find lowest energy structure for each composition
    if filtered_results:
        # Group by formula and find minimum energy
        formula_groups = {}
        for result in filtered_results:
            formula = result['formula']
            if formula not in formula_groups:
                formula_groups[formula] = result
            elif result['energy_per_atom'] < formula_groups[formula]['energy_per_atom']:
                formula_groups[formula] = result
        
        lowest_energy_structures = list(formula_groups.values())
        
        print(f"\nMost stable structure for each composition ({len(lowest_energy_structures)}):")
        for structure in lowest_energy_structures:
            print(f"  {structure['formula']}: {structure['energy_per_atom']:.3f} eV/atom, "
                  f"{structure['space_group_symbol']} (entry_id: {structure['entry_id']})")
        
        # Save filtered most stable structures
        with open("most_stable_structures.json", 'w') as f:
            json.dump(lowest_energy_structures, f, indent=2)
        print("Most stable structures saved to: most_stable_structures.json")
    
    return filtered_results

# Usage example
if __name__ == "__main__":
    # Set root directory of calculation folders
    root_dir = "./"  # Modify to your actual path
    
    # Batch extract results
    complete_results, summary_results = batch_extract_vasp_results(root_dir)
    
    # If there are results, perform further analysis
    if summary_results:
        # Display first few results
        print("\n==========Results==========")
        for i, result in enumerate(summary_results):
            print(f"{i+1:2d}. Entry: {result['entry_id']:<15} "
                  f"Formula: {result['formula']:<10} "
                  f"E={result['final_energy']:8.3f} eV "
                  f"E/atom={result['energy_per_atom']:7.3f} eV/atom "
                  f"SG={result['space_group_symbol']:<10} "
                  f"Conv={'Yes' if result['converged'] else 'No'}")
    
    # Example: Load complete data later
    # complete_entries = load_vasp_entries("entry.pkl")
    # print(f"Loaded {len(complete_entries)} complete ComputedEntry objects")
