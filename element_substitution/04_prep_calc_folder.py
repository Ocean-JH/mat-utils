#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
        Step: 04
        Create calculation directory in batches
        Author: Jianghai@BUAA
"""
import os
import shutil


# Backup raw_folder
ini_path = r"~\ElementSub"
main_work_path = r"~\ElementSub_Calc"
shutil.copytree(ini_path, main_work_path)


dirs = os.listdir(main_work_path)

for dir in dirs:
    workdir = os.path.join(main_work_path, dir)
    files = os.listdir(workdir)

    for file in files:
        fileName = file.split("-")
        Calc_path = os.path.join(workdir, "{}-{}-{}".format(fileName[0], fileName[1], fileName[2]))
        os.mkdir(Calc_path)
        shutil.move(os.path.join(workdir, file), os.path.join(Calc_path, "POSCAR"))
