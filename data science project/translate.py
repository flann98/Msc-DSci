# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:13:04 2025

@author: brend
"""

import gzip;
# Requires 'biopython' module be installed.
from Bio.PDB.MMCIF2Dict import MMCIF2Dict;
import numpy as np;
import os;
import pandas as pd;

def extract(ipt):
    """
    This function extracts the desired dataset files to a directory to be used
    by the translate function found below.
    
    Parameters
    ----------
    ipt : File due to be extracted and exported to directory.

    Returns
    -------
    File information to be translated for CNN application.

    """
    
    # Opens gzip file to create a dictionary for the CIF file.
    with gzip.open(ipt, 'rt') as file:
        print(file);
        cifDict = MMCIF2Dict(file);
        print(cifDict);
        df = pd.DataFrame.from_dict(cifDict, orient='index');
        return df;
        
def check_file(file):
    """
    This file checks each of the proteins in the dataset and returns the
    largest dimension of either x, y or z coordinates.

    Parameters
    ----------
    file: File for which the the furthest distance is to be checked.

    Returns
    -------
    The largest dimension's max distance from (0, 0, 0) 
    for the protein structure on any axis.

    """
    
    # Performs extract function for each file in the dataset folder.
    data = extract(file);
    dataframe = data.loc[['_atom_site.type_symbol',
                          '_pdbx_poly_seq_scheme.mon_id',
                          '_atom_site.Cartn_x', 
                          '_atom_site.Cartn_y', 
                          '_atom_site.Cartn_z']];
    # Splitting the list's values out into columns.
    dataframe = pd.DataFrame(dataframe[0].tolist(), index=dataframe.index);
    Xmax = np.max(np.sqrt(dataframe.loc['_atom_site.Cartn_x']
                          .astype('float64').values ** 2));
    Ymax = np.max(np.sqrt(dataframe.loc['_atom_site.Cartn_y']
                          .astype('float64').values ** 2));
# =============================================================================
#     Zmax = np.max(np.sqrt(dataframe.loc['_atom_site.Cartn_z']
#                           .astype('float64').values ** 2));
# =============================================================================
    # Returns the greatest cartesian protrusion from (0, 0, 0) on any axis.
    return np.max([Xmax, Ymax]), dataframe;

def translate(folder):
    """
    This function takes .cif files and extracts the desired data from said
    files for the means of machine learning modelling.

    Parameters
    ----------
    ipt : .cif file to be read and truncated.

    Returns
    -------
    .cif file data as a grid for later utilisation by both Bayesian
    and conventional CNN models.

    """
    
    # Creating a dictionary for atomic types and corresponding grid placement.
    atom_types = {'H' : 1,
                  'C' : 2,
                  'O' : 3,
                  'N' : 4,
                  'S' : 5};
    
    # Initialising the size of the 3D grid structure necessary for convolution.
    gridSize = 0;
    # Performs check_file function for each file in the dataset folder.
    for file in os.listdir(folder):
        if '.cif.gz' in file:
            # Only the grid size update check is needed here.
            updtSize, _ = check_file(os.path.join(folder, file));
            if updtSize > gridSize:
                gridSize = updtSize;
    # The grid needs to be the size of the furthest distance in all
    # directions, with an additional space granted for (0, 0, 0).
    double_gridSize = int(gridSize * 1000) * 2 + 1;
    
    # Performs extract function for each file in the dataset folder.
    for file in os.listdir(folder):
        if '.cif.gz' in file:
            # Only the data is required for this step.
            _, data = check_file(file);
            # Initialise new working grid of NaN values; empty space.
            work_grid = np.empty((double_gridSize, double_gridSize));
            work_grid[:] = np.nan;
            # Creating the labels to be saved corresponding to the grid.
            work_label = data.loc[['_atom_site.type_symbol',
                                  '_pdbx_poly_seq_scheme.mon_id'],:];
            # Mapping the atomic z-depth to the atoms' positions in the grid.
            for atom in data.columns:
                X_grid = data.loc['_atom_site.Cartn_x', atom] 
                + (gridSize * 1000) + 1;
                Y_grid = data.loc['_atom_site.Cartn_y', atom] 
                + (gridSize * 1000) + 1;
                # If the value is a NaN the 5 length array is created.
                if work_grid[X_grid, Y_grid] == np.nan:
                    work_grid[X_grid, Y_grid] = np.zeros(5);
                    work_grid[X_grid, Y_grid][:] = np.nan;
                    # Adding the z-position of the atom and amino acid attrib.
                    work_grid[[X_grid, Y_grid] \
                        [atom_types[data.loc['_atom_site.type_symbol', atom]]]] \
                        = [data.loc['_pdbx_poly_seq_scheme.mon_id', atom], \
                           data.loc['_atom_site.Cartn_z', atom]];
                # If present the array is amended in the position for the atom.
                else:
                    work_grid[[X_grid, Y_grid] \
                        [atom_types[data.loc['_atom_site.type_symbol', atom]]]] \
                        = [data.loc['_pdbx_poly_seq_scheme.mon_id', atom], \
                           data.loc['_atom_site.Cartn_z', atom]];
            # Saves the grid 'image' data to 3d/'file'.csv
            np.savetxt(f'3d/{file}.csv', work_grid, delimiter=',');
            # Saves the grid's labelling amino acid sequence.
            np.savetxt(f'3d/{file}-label.csv', work_label, delimiter=',');
    return;