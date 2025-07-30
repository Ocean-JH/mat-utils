#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-07-22
Description:
    Visualizes a property distribution across the periodic table.
"""

import os
import json
from typing import List, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors


class PeriodicTableMapper:
    """
        A class to map and visualize elemental properties onto a periodic table layout.
    """

    def __init__(self,
                 cell_size: float = 1.0,
                 cell_interval: float = 0.1,
                 cell_edge: float = 0.5,
                 cmap: str = 'YlOrRd',
                 layout_path: str = 'periodic_table_layout.json'):
        """
        Initialize the mapper with visual parameters and layout.

        Parameters:
            cell_size (float): Size of each cell.
            cell_interval (float): Interval between cells.
            cell_edge (float): Thickness of cell borders.
            cmap (str): Matplotlib colormap name.
            layout_path (str): Path to the JSON layout file.
        """
        self.cell_size = cell_size
        self.cell_interval = cell_interval
        self.cell_edge = cell_edge
        self.cmap = cmap
        self.data = None
        self._normalize = False
        self.layout = self._load_layout(layout_path)

    @staticmethod
    def _load_layout(path: str) -> dict:
        """
        Load periodic table layout from JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Layout file not found: {path}")
        with open(path, 'r') as file:
            return json.load(file)

    def _construct_table(self, data_dict: dict) -> list:
        """
        Construct the table containing elements and their plot positions.
        """
        table = []
        for element, info in self.layout.items():
            z = info['Z']
            row, col = self._position(z, info['period'], info['group'])

            val = data_dict.get(element, np.nan)
            table.append([z, element, col, row, val])

        # Dummy placeholders for Lanthanides and Actinides
        table.extend(
            [
                [None, 'LA', 3, 6, np.nan],
                [None, 'AC', 3, 7, np.nan],
                [None, 'LA', 2, 8, np.nan],
                [None, 'AC', 2, 9, np.nan],
            ]
        )

        return table

    @staticmethod
    def _position(z: int, period: int, group: int) -> tuple:
        """
        Determine row and column on the periodic table for an element.
        """
        if 57 <= z <= 71:  # Lanthanides
            return 8, z - 57 + 3
        elif 89 <= z <= 103:  # Actinides
            return 9, z - 89 + 3
        return period, group

    @staticmethod
    def _detect_element_column(data: pd.DataFrame) -> Union[str, None]:
        """
        Detect the column or index containing element symbols.
        """
        if data.index.dtype == 'object':
            index_values = data.index.dropna().unique()
            if all(isinstance(v, str) and 1 <= len(v) <= 3 for v in index_values[:10]):
                return 'index'

        for col in data.columns:
            if data[col].dtype == 'object':
                values = data[col].dropna().unique()
                if all(isinstance(v, str) and 1 <= len(v) <= 3 for v in values[:10]):
                    return col

        return None

    def set_data(self, data: Union[dict, pd.Series, pd.DataFrame], normalize: bool = True):
        """
        Set the property data for visualization.
        """
        if isinstance(data, dict):
            df = pd.DataFrame(list(data.items()), columns=['element', 'value']).set_index('element')
        elif isinstance(data, pd.Series):
            df = data.to_frame(name=data.name or 'value')
        elif isinstance(data, pd.DataFrame):
            ele_col = self._detect_element_column(data)
            if ele_col is None:
                raise ValueError("Unable to detect the element column.")

            df = data.copy() if ele_col == 'index' else data.set_index(ele_col)
        else:
            raise TypeError("Unsupported data type. Data must be a dict, pandas Series, or pandas DataFrame.")

        self.data = df.div(df.sum(axis=1), axis=0).fillna(0) if normalize else df
        self._normalize = normalize

    def plot(self, title: str = 'Periodic Table Distribution', attribute: str = None, save_path: str = None):
        """
        Plot a single attribute on the periodic table.
        """
        if self.data is None or self.data.empty:
            raise ValueError("Data not set. Use set_data() first.")

        if attribute is None:
            if self.data.shape[1] == 1:
                attribute = self.data.columns[0]
            else:
                raise ValueError(
                    f"Multiple columns detected. Please specify an attribute.\nAvailable attributes: {self.data.columns.tolist()}")

        if attribute not in self.data.columns:
            raise ValueError(
                f"Attribute '{attribute}' not found in data.\nAvailable attributes: {self.data.columns.tolist()}")

        data_dict = self.data[attribute].dropna().to_dict()
        full_title = f"{title} - {attribute}" if attribute != 'value' else title
        self._draw(data_dict, full_title, save_path)

    def plot_attributes(self, attributes: List[str] = None, save: bool = True):
        """
        Plot multiple attributes as separate heatmaps.
        """
        if self.data is None or self.data.empty:
            raise ValueError("Data not set. Use set_data() first.")

        attrs = attributes or self.data.columns.tolist()
        valid_attrs = [a for a in attrs if a in self.data.columns]

        for attr in valid_attrs:
            save_path = f"attributes/{attr}_periodic_table_distribution.png" if save else None
            if save:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.plot(attribute=attr, save_path=save_path)

    def _draw(self, data_dict: dict, title: str, save_path: str):
        """
        Internal drawing method using matplotlib.
        """
        table = self._construct_table(data_dict)

        fig, ax = plt.subplots(figsize=(18, 9))
        cmap = plt.get_cmap(self.cmap)
        cmap.set_under('None')

        values = [v for *_, v in table if not np.isnan(v)]
        norm = colors.Normalize(vmin=min(values), vmax=max(values))
        cm_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(cm_mappable, ax=ax)

        for z, symbol, col, row, val in table:
            # Compute x, y coordinates
            x = (self.cell_size + self.cell_interval) * (col - 1)
            y = 11 - (self.cell_size + self.cell_interval) * row - (self.cell_size * 0.5 if row >= 8 else 0)

            if z is not None:
                # Draw cell
                face_color = cmap(norm(val)) if not np.isnan(val) else 'white'
                rect = plt.Rectangle((x, y),
                                     self.cell_size,
                                     self.cell_size,
                                     lw=self.cell_edge,
                                     edgecolor='k',
                                     facecolor=face_color
                                     )
                ax.add_patch(rect)

            # Atomic number
            ax.text(x + 0.04, y + 0.8, str(z) if z is not None else '',
                    ha='left', va='center',
                    fontsize=6, family='Helvetica')
            # Element symbol
            ax.text(x + 0.5, y + 0.5, symbol,
                    ha='center', va='center',
                    fontsize=9, weight='bold', family='Helvetica')

            # Value display
            if val is not None and not np.isnan(val) and val > 0:
                text = f"{val:.2%}" if self._normalize else val
                ax.text(x + 0.5, y + 0.12, text,
                        ha='center', va='center',
                        fontsize=7, family='Helvetica')

        ax.set_title(title, fontsize=16)
        plt.axis('off')
        plt.axis('equal')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    contingency = pd.read_csv("example.csv", index_col=0)
    mapper = PeriodicTableMapper()
    mapper.set_data(contingency, normalize=False)
    mapper.plot_attributes(save=True)
