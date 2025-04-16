#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
        Step: 05
        Data processing
        Author: Jianghai@BUAA
"""
import os
import json
import linecache
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Oszicar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def check_convergence(calc_dir):
    file = os.path.join(calc_dir, 'OUTCAR')
    with open(file, 'r') as f:
        content = f.read()
        return 'reached required accuracy' in content


def data_extraction(root_dir=None, depth=2):
    if root_dir is None:
        root_dir = r'~'

    data_info = pd.DataFrame(columns=['group',
                                      'parent',
                                      'Ge', 'Sb', 'Te',
                                      'space_group_symbol',
                                      'space_group_number',
                                      'convergence',
                                      'energy_per_atom',
                                      'formation_energy_per_atom'])

    for root, dirs, files in os.walk(root_dir):
        cur_depth = root.count(os.sep) - root_dir.count(os.sep)
        if cur_depth == depth:
            if 'OUTCAR' in files and 'OSZICAR' in files and 'vasprun.xml' in files:
                group = root.split('\\')[-2]
                parent = root.split('\\')[-1]
                convergence = check_convergence(root)
                species = linecache.getline(os.path.join(root, 'POSCAR'), 6).split()
                stoichiometry = linecache.getline(os.path.join(root, 'POSCAR'), 7).split()
                composition = {'Ge': '0',
                               'Sb': '0',
                               'Te': '0'}
                composition.update({species[i]: stoichiometry[i] for i in range(len(species))})

                xml_path = os.path.join(root, 'vasprun.xml')
                oszicar_path = os.path.join(root, 'OSZICAR')
                try:
                    dataset = Vasprun(xml_path)

                    with open(os.path.join(os.path.dirname(root_dir), 'EleSub_dataset.json'), 'a',
                              newline='\n') as file:
                        json.dump(dataset.as_dict(), file)
                        file.write('\n')

                    n_atoms = len(dataset.ionic_steps[0]["structure"])
                    energy = Oszicar(oszicar_path).final_energy
                    energy_per_atom = energy / n_atoms

                    ge_energy_per_atom = -4.518072
                    sb_energy_per_atom = -4.141693
                    te_energy_per_atom = -3.141757

                    formation_energy_per_atom = (energy - (int(composition['Ge']) * ge_energy_per_atom +
                                                           int(composition['Sb']) * sb_energy_per_atom +
                                                           int(composition['Te']) * te_energy_per_atom)) / (
                                                        int(composition['Ge']) + int(composition['Sb']) + int(
                                                    composition['Te']))

                    structure = Structure.from_file(os.path.join(root, 'CONTCAR'))
                    spa = SpacegroupAnalyzer(structure)
                    spg_symbol = spa.get_space_group_symbol()
                    spg_number = spa.get_space_group_number()

                    data = {'group': group,
                            'parent': parent,
                            'Ge': int(composition['Ge']),
                            'Sb': int(composition['Sb']),
                            'Te': int(composition['Te']),
                            'space_group_symbol': spg_symbol,
                            'space_group_number': spg_number,
                            'convergence': convergence,
                            'energy_per_atom': energy_per_atom,
                            'formation_energy_per_atom': formation_energy_per_atom}

                    data = pd.DataFrame(data, index=[0])
                    data_info = pd.concat([data_info, data], ignore_index=True)
                except:
                    print('Error in {}, skipping...'.format(root))
                    continue

    data_info.to_csv(os.path.join(os.path.dirname(root_dir), 'EleSub_dataset.csv'), index=False)


def data_processing(data_path):
    data_info = pd.read_csv(data_path)
    data_info['group'] = data_info['group'].astype(str)
    data_info['convergence'] = data_info['convergence'].astype(str)
    data_info = data_info[data_info['convergence'].str.contains('True')]
    data_info.sort_values(by=['formation_energy_per_atom'], inplace=True)

    data_info[['Ge', 'Sb', 'Te']] = data_info[['Ge', 'Sb', 'Te']].astype(int)
    data_info['m'] = data_info['Ge']
    data_info['n'] = data_info['Sb'] / 2
    data_info['n'] = data_info['n'].astype(int)
    data_info['x'] = data_info['m'] / (data_info['m'] + data_info['n'])
    data_info['y'] = data_info['formation_energy_per_atom']

    data_info.to_csv(r'~\valid_data.csv', index=False)


def select_data(data_path):
    data_info = pd.read_csv(data_path)
    data_info = data_info[data_info['Ehull'] < 0.13]
    data_info = data_info[data_info['formation_energy_per_atom'] < 0]
    data_info = data_info[data_info['space_group_number'] != 1]
    data_info = data_info[~data_info['group'].str.contains('1-2-4')]
    data_info = data_info[~data_info['group'].str.contains('1-4-7')]
    data_info = data_info[~data_info['group'].str.contains('1-6-10')]
    data_info = data_info[~data_info['group'].str.contains('2-2-5')]
    data_info = data_info[~data_info['group'].str.contains('2-2-5_Permutation')]

    data_info.to_csv(r'~\EleSub_selected.csv', index=False)


def convex_hull(data_path):
    data_info = pd.read_csv(data_path)
    data = data_info[['x', 'y']]
    edge = [[0, -0.1202678],
            [1, -0.0942086]]
    data = data._append(pd.DataFrame(edge, columns=data.columns))
    data.reset_index(inplace=True, drop=True)
    pts = np.array(data)

    '''
                Convex hull determination
    '''
    hull = ConvexHull(pts)

    '''
                Set of convex hull vertices
    '''
    hull_vts_all = []
    for pt in hull.vertices:
        vertex = pts[pt]
        hull_vts_all.append(vertex)
    hull_vts_all = pd.DataFrame(hull_vts_all)

    hull_vts_negative = hull_vts_all[~(hull_vts_all[[1]] > 0).any(axis=1)]

    '''
                Equation determination
    '''
    HULL = ConvexHull(hull_vts_negative)

    HULL_Eqs = pd.DataFrame(HULL.equations)
    HULL_Eqs = np.array(HULL_Eqs)
    HULL_Eqs = np.delete(HULL_Eqs, [0], axis=0)

    '''
                Calculation of E_hull
    '''
    a, b, c, x, y = [], [], [], [], []

    for i in range(len(HULL_Eqs)):
        a_i = HULL_Eqs[i][0]
        a.append(a_i)
        b_i = HULL_Eqs[i][1]
        b.append(b_i)
        c_i = HULL_Eqs[i][2]
        c.append(c_i)

    for j in range(len(pts)):
        x_j = pts[j][0]
        x.append(x_j)
        y_j = pts[j][1]
        y.append(y_j)

    e_on_hull_set = []
    for i in range(len(pts)):
        for j in range(len(HULL_Eqs)):
            e_on_hull = -(x[i] * a[j] + c[j]) / b[j]
            e_on_hull_set.append(e_on_hull)
    e_on_hull_set = pd.DataFrame(np.array(e_on_hull_set).reshape(len(pts), len(HULL_Eqs)))
    e_on_hull = e_on_hull_set.max(axis=1)  # Select the maximum value as convex hull energy

    eform = data_info['y']
    ehull_set = []
    for i in range(len(eform)):
        ehull = eform[i] - e_on_hull[i]  # Calculation of E_hull
        ehull_set.append(ehull)

    '''
                File output
    '''
    data_info['Ehull'] = ehull_set
    data_info = pd.DataFrame(data_info)
    data_info.to_csv(r'~\EleSub_Energy_above_Hull.csv', index=False)


def plot_convex_hull(data_path):
    data = pd.read_csv(data_path)
    x1 = data['x']
    y1 = data['y']

    seeds_path = r'F:\Dataset\mp_GST_Data.csv'
    seeds = pd.read_csv(seeds_path)
    seeds = seeds[seeds['Ge'] + seeds['Sb'] / 2 * 3 == seeds['Te']]
    seeds['convergence'] = seeds['convergence'].astype(str)
    seeds = seeds[seeds['convergence'].str.contains('True')]
    seeds = seeds[['Ge', 'Sb', 'Te', 'space_group_symbol', 'space_group_number', 'formation_energy_per_atom']]
    seeds[['Ge', 'Sb', 'Te']] = seeds[['Ge', 'Sb', 'Te']].astype(int)
    seeds['m'] = seeds['Ge']
    seeds['n'] = seeds['Sb'] / 2
    seeds['n'] = seeds['n'].astype(int)
    seeds['x'] = seeds['m'] / (seeds['m'] + seeds['n'])
    seeds['y'] = seeds['formation_energy_per_atom']

    # Metastable cubic structure
    cubic_data = np.array([[0, -0.02081418],
                           [0.333333, -0.07518477],
                           [0.5, -0.052255621],
                           [0.666667, -0.071077905],
                           [1, -0.075879095]])

    x2 = seeds['x']
    y2 = seeds['y']

    hull_data = data[['x', 'y']]
    hull_seeds = seeds[['x', 'y']]
    hull_data = pd.concat([hull_data, hull_seeds])
    hull_data = hull_data[~(hull_data[['y']] > 0).any(axis=1)]

    hull_data = np.array(hull_data)
    hull = ConvexHull(hull_data)

    simplices = hull.simplices
    simplices = np.delete(simplices, [0, 1, 2, 4, 5, 6, 7, 8], axis=0)

    plt.figure(figsize=(16, 9))
    ax = plt.subplot(111)
    plt.scatter(x1, y1, color='None', marker='d', edgecolor='teal', s=100, alpha=0.25, label='Elemental Substitution')
    plt.scatter(x2, y2, color='sienna', marker='^', s=100, label='Materials Project')
    plt.scatter([0.25, 0.33333333, 0.5],
                [-0.125695, -0.1247125, -0.11877529],
                color='red', marker='d', s=100, label='On Convex Hull')
    plt.scatter(cubic_data[:, 0], cubic_data[:, 1], color='red', marker='s', s=100, label='Cubic')

    plt.rc('font', family='Times New Roman')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("${m}/{m + n}$", fontsize=18)
    plt.ylabel("Formation Energy (eV / atom)", fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.3)

    ax.set_xticks([0.142857, 0.25, 0.333333, 0.5, 0.666667, 0.75])
    ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25])

    plt.legend(['Elemental Substitution', 'Materials Project', 'On Convex Hull', 'Cubic'], prop={'size': 18})

    plt.text(0.96, -0.118, 'GeTe', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.text(0.22, -0.15, '$\mathregular{Ge_1Sb_6Te_{10}}$', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.text(0.305, -0.15, '$\mathregular{Ge_1Sb_4Te_7}$', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.text(0.305, -0.15, '$\mathregular{Ge_1Sb_4Te_7}$', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.text(0.47, -0.145, '$\mathregular{Ge_1Sb_2Te_4}$', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.text(0.635, -0.135, '$\mathregular{Ge_2Sb_2Te_5}$', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.text(0.005, -0.138, '$\mathregular{Sb_2Te_3}$', fontdict={'family': 'Times New Roman', 'size': 15})

    for i in simplices:
        # print(hull_data[i, 0])
        # print(hull_data[i, 1])
        plt.plot(hull_data[i, 0], hull_data[i, 1], c='gold', linewidth=3)

    for i in range(len(cubic_data) - 1):
        plt.plot(cubic_data[[i, i + 1], 0], cubic_data[[i, i + 1], 1], c='gold', linewidth=3)

    plt.savefig('Elemental_Substitution_Convex_Hull.png', dpi=1200)
    plt.show()


if __name__ == '__main__':
    path = r'~'
    file = os.path.join(path, r'valid_data.csv')
    data_extraction()
    data_processing(os.path.join(path, r'EleSub_dataset.csv'))
    convex_hull(os.path.join(path, r'valid_data.csv'))
    select_data(os.path.join(path, r'EleSub_Energy_above_Hull.csv'))
    plot_convex_hull(file)
