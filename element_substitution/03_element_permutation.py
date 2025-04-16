#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
            Step: 03
            Permutation of elements with the same stoichiometry (i.e., '2-2-5')
            Author: Jianghai@BUAA
"""
import os


dirs = os.listdir(r'~\ElementSub')

for dir in dirs:
    composition = dir.split("-")
    # Check for duplicate elements
    if len(composition) != len(set(composition)):
        # Access to the directory of permutation files
        os.mkdir(r'~\ElementSub\{}-{}-{}_Permutation'.format(*composition))
        files = os.listdir(r'~\ElementSub\\' + dir)

        for i in range(len(files)):
            fileName = r'~\ElementSub\\' + dir + '\\' + files[i]
            file_ini = open(fileName, "r")
            content = file_ini.read()
            file_ini.close()

            with open(r'~\ElementSub\{}-{}-{}_Permutation'.format(*composition) + '\\' + '{}_permutation'.format(files[i]), "w") as file_permutated:
                content = content.replace('Ge', 'G_e').replace('Sb', 'Ge').replace('G_e', 'Sb')
                file_permutated.write(content)
                file_permutated.close()
