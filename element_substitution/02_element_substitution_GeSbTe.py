#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
            Step: 02
            Batch element substitution for POSCAR files
            Replace elements with Ge-Sb-Te
            Author: Jianghai@BUAA
"""
import os
import math
import linecache


def seq_element(stoichiometry, comp):
    """
            Sort the elements by stoichiometry
    """
    element_seq = []
    for idx in range(3):
        for key, value in stoichiometry.items():
            if value == comp[idx]:
                element_seq.append(key)
                stoichiometry.update({key: '0'})
                break
    return element_seq


# Access to the directory of raw files
dirs = os.listdir(r'~\Stoichiometry')
# Create directory of destination files
os.mkdir(r'~\ElementSub')
for dir in dirs:
    # Create subdirectories of destination files
    os.mkdir(r'~\ElementSub\{}'.format(dir))
    # Access to raw filenames
    files = os.listdir(r'~\Stoichiometry\\' + dir)

    for i in range(len(files)):
        # Concatenate to get filenames
        fileName = r'~\Stoichiometry\\' + dir + '\\' + files[i]

        # Bind target elements to the corresponding stoichiometry
        element = ['Ge', 'Sb', 'Te']
        composition = dir.split("-")
        stoichiometry = {element[i]: composition[i] for i in range(3)}

        # Get stoichiometry of raw files and reduce it
        comp = linecache.getline(fileName, 7).split()
        comp_int = [int(i) for i in comp]
        gcd = math.gcd(comp_int[0], comp_int[1], comp_int[2])         # Calculate the greatest common divisor (GCD)
        if gcd != 1:
            comp_int = [int(i / gcd) for i in comp_int]
            comp = [str(i) for i in comp_int]
        # Get the elements sorted correctly
        seq_ele = seq_element(stoichiometry, comp)

        # Copy the contents of raw files
        file_ini = open(fileName, "r")
        content = file_ini.read()
        file_ini.close()

        # Modify the corresponding elements and write to destination files
        with open(r'~\ElementSub\\' + dir + '\\' + '{}-{}-POSCAR'.format(files[i].split("_")[0], files[i].split("_")[1].split(".")[0]), "w") as file_processed:
            ini_ele = linecache.getline(fileName, 6).split()
            temp_ele = ['!!!', '@@@', '###']
            # Corresponding elements substitution
            for x in range(3):
                # Avoid interference
                if len(ini_ele[x]) == 2:
                    content = content.replace('{}'.format(ini_ele[x]), '{}'.format(temp_ele[x]))
            for y in range(3):
                content = content.replace('{}'.format(ini_ele[y]), '{}'.format(temp_ele[y]))
            for z in range(3):
                content = content.replace('{}'.format(temp_ele[z]), '{}'.format(seq_ele[z]))
            file_processed.write(content)
            file_processed.close()
