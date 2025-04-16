#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Author: Jianghai@BUAA 2024
"""

import io
import os
import sys
import shutil
import warnings
import linecache

import numpy as np


def prepare_dir(file='POSCAR', stretch_distances=None):
    """
            Create file directories of structure file for first principles tensile test.

    @type file: str
    @type stretch_distances: List
    :param file: File_name of the model in 'POSCAR' format
    :param stretch_distances: List of  Stretch distances
    """

    # file_dir = os.path.join(os.getcwd(), file)

    if stretch_distances is None:
        stretch_distances = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 9.0]

    with open(file, 'r') as f:
        raw_content = f.read()
        line_num = sum(1 for _ in f)

    calc_num = len(stretch_distances)

    calc_dir = os.path.join(os.getcwd(), 'Calc_dir')
    if not os.path.exists(calc_dir):
        os.mkdir(calc_dir)

    if not os.path.exists(os.path.join(calc_dir, 'ini_structure')):
        os.mkdir(os.path.join(calc_dir, 'ini_structure'))
    shutil.copy(file, os.path.join(calc_dir, 'ini_structure'))

    for i in range(calc_num):
        work_dir = os.path.join(calc_dir, 'Distance_{}'.format(stretch_distances[i]))
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)

        processed_content = raw_content
        c_vector_content = linecache.getline('POSCAR', 5)
        c_vector = linecache.getline('POSCAR', 5).split()
        c_z = float(c_vector[2], ) + stretch_distances[i]
        processed_content = processed_content.replace(c_vector_content,
                                                      '        {}         {}         {}\n'
                                                      .format(c_vector[0], c_vector[1], c_z, '.10f'))

        for row in range(9, line_num):
            coord_content = linecache.getline('POSCAR', row)
            coord = linecache.getline('POSCAR', row).split()
            if float(coord[2]) > float(c_vector[2]) / 2 + 0.5:
                coord_z = float(coord[2]) + stretch_distances[i]

                processed_content = processed_content.replace(coord_content,
                                                              '     {}         {}        {}\n'
                                                              .format(coord[0], coord[1], coord_z, '.9f'))

        with open(os.path.join(work_dir, 'POSCAR'), 'w') as f:
            f.write(processed_content)

        print('Distance: {} created.'.format(stretch_distances[i]))


if __name__ == '__main__':
    print("""
    ___  _             _       ___        _               _         _              _____                   _  _         _____            _   
   / __\(_) _ __  ___ | |_    / _ \ _ __ (_) _ __    ___ (_) _ __  | |  ___  ___  /__   \ ___  _ __   ___ (_)| |  ___  /__   \ ___  ___ | |_ 
  / _\  | || '__|/ __|| __|  / /_)/| '__|| || '_ \  / __|| || '_ \ | | / _ \/ __|   / /\// _ \| '_ \ / __|| || | / _ \   / /\// _ \/ __|| __|
 / /    | || |   \__ \| |_  / ___/ | |   | || | | || (__ | || |_) || ||  __/\__ \  / /  |  __/| | | |\__ \| || ||  __/  / /  |  __/\__ \| |_ 
 \/     |_||_|   |___/ \__| \/     |_|   |_||_| |_| \___||_|| .__/ |_| \___||___/  \/    \___||_| |_||___/|_||_| \___|  \/    \___||___/ \__|
                                                            |_|                                                                              
    
    Author: Jianghai@BUAA 2024
    Creating calculation directories...
    """)

    prepare_dir()

    print('All directories prepared.\n')
    warnings.warn("Remember to double check before submit!!!", UserWarning)
