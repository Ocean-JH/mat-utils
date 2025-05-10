#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
uspex_hull_analysis.py

This script analyzes structure data from USPEX and/or external sources (e.g., Materials Project),
performs convex hull construction in ternary composition space, and visualizes the results.

Main functionalities:
- Parse USPEX extended convex hull data and compute derived formation energies.
- Project compositions onto a 2D ternary coordinate system.
- Construct 3D convex hulls to determine thermodynamic stability (Ehull).
- Visualize ternary convex hull diagrams with optional seeds/reference points.

Dependencies:
- numpy
- pandas
- matplotlib
- scipy

Author: Wang Jianghai @BUAA
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class DataAnalysis:
    """Analyze structure data and visualize convex hull diagram in ternary space."""
    def __init__(self, path):
        self.path = path

    @property
    def structure_info(self):
        """
        Load USPEX structure data in the order of decreasing stability and compute reference formation energy.
        Returns:
            pd.DataFrame: Structure info with ref_eform and triangle coordinates.
        """
        file = os.path.join(self.path, 'extended_convex_hull')
        with open(file, 'r') as f:
            lines = f.readlines()[6:]

        df = pd.DataFrame(lines)
        df = df[0].str.split(expand=True)
        df = df.drop(df.columns[[1, 5]], axis=1)
        df.columns = ['ID', 'Ge', 'Sb', 'Te', 'Enthalpy', 'Volume', 'Fitness', 'SYMM', 'X1', 'X2', 'Y']
        structure_info = df.apply(pd.to_numeric, errors='coerce')

        # Calculate reference formation energy per atom
        ref_e = {'Ge': -4.51807214, 'Sb': -4.141692635, 'Te': -3.141756903}
        structure_info['ref_eform'] = structure_info.apply(lambda x: (x['Enthalpy'] * (x['Ge'] + x['Sb'] + x['Te']) - (
                    x['Ge'] * ref_e['Ge'] + x['Sb'] * ref_e['Sb'] + x['Te'] * ref_e['Te'])) / (
                                                                       x['Ge'] + x['Sb'] + x['Te']), axis=1)

        return structure_info

    @property
    def seeds_info(self):
        """
                Load seed data from Materials Project.
                Returns:
                    pd.DataFrame: Seed structures with composition and formation energy.
        """
        # file = r'seeds_info.csv'
        file = r'mp_GST_Data.csv'

        return pd.read_csv(file)

    def _comp_triangle(self, info):
        """
                Convert elemental compositions to 2D ternary coordinates (X1, X2).
                Args:
                    source (str): One of 'uspex', 'uspex_ref', 'seeds'
                Returns:
                    pd.DataFrame: Data with Ge, Sb, Te, X1, X2, Y
        """
        if info == 'uspex':
            comp = self.structure_info[['Ge', 'Sb', 'Te', 'Y']]
            comp['sum'] = comp.iloc[:, 0:3].sum(axis=1)

        elif info == 'uspex_ref':
            comp = self.structure_info[['Ge', 'Sb', 'Te', 'ref_eform']]
            comp.rename(columns={'ref_eform': 'Y'}, inplace=True)
            comp['sum'] = comp.iloc[:, 0:3].sum(axis=1)

        elif info == 'seeds':
            comp = self.structure_info[['Ge', 'Sb', 'Te', 'formation_energy_per_atom']]
            comp.rename(columns={'formation_energy_per_atom': 'Y'}, inplace=True)
            comp['sum'] = comp.iloc[:, 0:3].sum(axis=1)
        else:
            raise ValueError("info must be 'uspex', 'uspex_ref' or 'seeds', not {}".format(info))

        comp_frac_0 = comp['Ge'] / comp['sum']
        comp_frac_1 = comp['Sb'] / comp['sum']
        comp['X1'] = comp_frac_0 + comp_frac_1 / 2
        comp['X2'] = comp_frac_1 * np.sqrt(3) / 2

        return comp

    @property
    def calc_convex_hull(self):
        """
                Compute energy above convex hull (Ehull) for each structure.
                Returns:
                    pd.DataFrame: Structure info with Ehull column.
        """
        npdata = np.array(self.structure_info)
        hull_data = self.structure_info[['X1', 'X2', 'Y']]
        nphull = np.array(hull_data)
        hull = ConvexHull(hull_data)

        hull_vertices = []
        for index in hull.vertices:
            vertex = npdata[index]
            hull_vertices.append(vertex)
        hull_vertices = pd.DataFrame(hull_vertices)

        hull_neg_vertices = hull_vertices[(hull_vertices[[3]] <= 0).any(axis=1)]
        hull_neg_vertices = pd.DataFrame(hull_neg_vertices)
        hull_vertices = hull_neg_vertices[[8, 9, 10]]

        HULL = ConvexHull(hull_vertices)

        HULL_Eqs = pd.DataFrame(HULL.equations)
        HULL_Eqs = HULL_Eqs[~(HULL_Eqs[[3]] == 0).any(axis=1)]
        HULL_Eqs = np.array(HULL_Eqs)
        print(HULL_Eqs)

        a, b, c, d, x, y = [], [], [], [], [], []

        for i in range(len(HULL_Eqs)):
            a_i = HULL_Eqs[i][0]
            a.append(a_i)
            b_i = HULL_Eqs[i][1]
            b.append(b_i)
            c_i = HULL_Eqs[i][2]
            c.append(c_i)
            d_i = HULL_Eqs[i][3]
            d.append(d_i)

        for j in range(len(nphull)):
            x_j = nphull[j][0]
            x.append(x_j)
            y_j = nphull[j][1]
            y.append(y_j)

        convex_hull_set = []
        for i in range(len(nphull)):
            for j in range(len(HULL_Eqs)):
                convex_hull = -(x[i] * a[j] + y[i] * b[j] + d[j]) / c[j]
                convex_hull_set.append(convex_hull)
        convex_hull_set = pd.DataFrame(np.array(convex_hull_set).reshape(len(nphull), len(HULL_Eqs)))

        convex_hull_set[convex_hull_set > 0] = -100  # Ignore point where convex hull > 0
        E_max = convex_hull_set.max(axis=1)  # Select the maximum value as convex hull energy

        Ef = hull_data['Y']
        E_hull = []
        for i in range(len(Ef)):
            Ehull = Ef[i] - E_max[i]  # Calculation of E_hull
            E_hull.append(Ehull)

        data = pd.DataFrame(self.structure_info)
        data['Ehull'] = E_hull

        return data

    def plot_convex_hull(self, seeds=True, ref=True):
        """
                Plot ternary convex hull diagram using 3D surface and optional seed data.
                Args:
                    seeds (bool): Include Materials Project seeds
                    ref (bool): Use reference formation energy
        """
        uspex_data = self._comp_triangle('uspex_ref' if ref else 'uspex')

        data = uspex_data[['X1', 'X2', 'Y']]
        data = data[~(data[['Y']] > 0.2).any(axis=1)]
        x = data['X1']
        y = data['X2']
        z = data['Y']

        hull_data = uspex_data[['Ge', 'Sb', 'Te', 'X1', 'X2', 'Y']]
        hull_data = hull_data[~(hull_data[['Y']] > 0).any(axis=1)]

        if ref :
            hull_data = hull_data[['X1', 'X2', 'Y']]
            corners = [0, 0, 0, 1, 0, 0, 0.5, 0.866, 0]
            corners = pd.DataFrame(np.array(corners).reshape(3, 3), columns=['X1', 'X2', 'Y'])
            hull_data = pd.concat([hull_data, corners])

        hull_data = hull_data[['X1', 'X2', 'Y']]
        hull_data = np.array(hull_data)
        hull = ConvexHull(hull_data)

        simplices = hull.simplices
        simplices = np.delete(simplices, 1, axis=0)

        plt.figure(figsize=(16, 9))
        # print(plt.style.available)
        plt.style.use('seaborn-colorblind')
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z)

        for i in simplices:
            hull_data[i[1], 1] += 0.0001
            ax.plot_trisurf(hull_data[i, 0], hull_data[i, 1], hull_data[i, 2],
                            color=(0.2745, 0.5098, 0.7059, 0), alpha=0.5)

        if seeds is True:
            seeds_data = self._comp_triangle('seeds')
            seeds_data = seeds_data[['X1', 'X2', 'Y']]
            seeds_data = seeds_data[~(seeds_data[['Y']] > 0.2).any(axis=1)]
            x_ = seeds_data['X1']
            y_ = seeds_data['X2']
            z_ = seeds_data['Y']

            seeds_hull_data = seeds_data[['X1', 'X2', 'Y']]
            seeds_hull_data = seeds_hull_data[~(seeds_hull_data[['Y']] > 0).any(axis=1)]
            seeds_hull_data = np.array(seeds_hull_data)
            seeds_hull = ConvexHull(seeds_hull_data)

            seeds_simplices = seeds_hull.simplices
            seeds_simplices = np.delete(seeds_simplices, 0, axis=0)

            ax.scatter(x_, y_, z_, c='r')

            for i in seeds_simplices:
                seeds_hull_data[i[1], 1] += 0.0001
                ax.plot_trisurf(seeds_hull_data[i, 0], seeds_hull_data[i, 1], seeds_hull_data[i, 2],
                                color=(1, 1, 0, 0), alpha=0.5)

        ax.set_title('Convex Hull Diagram', fontsize=20)
        # ax.set_zlabel('Formation Energy', fontsize=16)

        ax.grid(None)
        ax.axis('off')
        ax.set(xticklabels=[],
               yticklabels=[],
               zticklabels=[])
        ax.view_init(elev=10, azim=-95)

        # plt.savefig('ConvexHullDiagram.png', dpi=600)
        plt.show()


if __name__ == '__main__':
    file = r'results1'
    da = DataAnalysis(file)
    da.plot_convex_hull(seeds=True, ref=True)

    # data = da.structure_info
    # data.to_csv(r'USPEX_Structure_Analysis.csv', index=False)
