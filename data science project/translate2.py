# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 11:00:20 2025

@author: brend
"""

import os;
import gzip;
import shutil;
# Requires biopython installation.
from Bio.PDB import PDBParser;
import networkx as nx;
import numpy as np;
import csv;

def unzipper(ipt, opt):
    """
    Unzips the gzip files to separate directory .cif file format files.

    Parameters
    ----------
    ipt : File to be unzipped.
    opt : File directory for .cif file to be saved.

    Returns
    -------
    None.

    """
    # Checking location of output directory.
    os.makedirs(opt, exist_ok=True);
    
    # Fetching the base file name (without .gz extension).
    output_file_name = os.path.basename(ipt).replace('.gz', '');
    output_file_path = os.path.join(opt, output_file_name);
    
    try:
        # Open the .gz file and save the uncompressed file.
        with gzip.open(ipt, 'rb') as f_in:
            with open(output_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out);
        print(f"File unzipped and saved at: {output_file_path}");
    except Exception as e:
        print(f"An error occurred: {e}");
    return;
    
    
def unzip(folder_in, folder_out):
    """
    Calls the unzipper function to unzip each file in a directory.

    Parameters
    ----------
    folder_in : Folder containing desired files.
    folder_out : Directory for unzipped files to be saved in.

    Returns
    -------
    None.

    """
    
    for file in os.listdir(folder_in):
        unzipper(fr'{folder_in}\{file}', folder_out);
    return;

def parse(file, output_dir, threshold):
    """
    
    Parameters
    ----------
    file : .pdb file to be parsed.

    Returns
    -------
    atomic_coordinates : Cartesian coordinates of the atoms.
    atomic_types : Atomic types present in the protein.
    bonds : Bonds as neighbours to atoms within 3.0 of one another.

    """
    
    # Fetching the unique ID for the file to save the graphs under.
    fileID, _ = os.path.splitext(os.path.basename(file));
    
    try:
        # Parse the PDB file.
        parser = PDBParser();
        # Fetching the structure of the protein.
        structure = parser.get_structure('protein', file);
        
        # Create the GCN graph format.
        gcn_graph = nx.Graph();
        # Initialise the amino acid sequence.
        amino_acids = [];

        # Extract residues and atoms.
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Append current residue of amino acid sequence.
                    amino_acids.append(residue.resname);
                    res_id = (residue.id[1], chain.id);
                    # Looping to the atomic layer of the structure.
                    for atom in residue:
                        # Adding atom as node with pertinent info.
                        atom_id = (atom.id, 
                                   tuple(atom.coord), # Tuple to preserve info.
                                   atom.bfactor, 
                                   atom.occupancy);
                        atom_node = (atom_id, res_id);
                        gcn_graph.add_node(atom_node, atom=atom);
                        
                        # Add edges based on distance threshold.
                        add_edges = [];
                        for other_atm_id, other_atm in gcn_graph.nodes(data="atom"):
                            if other_atm_id[0] != atom_node:
                                dist = np.mean(atom_node[0][1]) - np.mean(other_atm_id[0][1]);
                                # Threshold for interaction.
                                if abs(dist) < threshold:
                                    add_edges.append([atom_node, other_atm_id, dist]);
                        for new in add_edges:
                            gcn_graph.add_edge(new[0], new[1], weight=new[2]);
        
        # Save the amino acid sequence as labels.
        with open(fr'{output_dir}/{fileID}-AAlabel.csv', 
                  mode='w', 
                  newline='') as fmt:
            for acid in amino_acids:
                fmt.write(f'{acid}\n');
        # Save the graph as adjacency list.
        nx.write_graph6(gcn_graph,
                         fr'{output_dir}/{fileID}-gcn_graph');
        return;

    except Exception as e:
        print(f"Error parsing PDB file: {e}");
        return None;

def iterator(direct=r'dataset/AlphaFold Protein Database e.coli/Uncompressed',
             output_dir=r'dataset/AlphaFold Protein Database e.coli/Uncompressed/graphs', 
             threshold=6.0):
    """
    

    Parameters
    ----------
    direct : input folder directory, optional
        DESCRIPTION. The default is 'dataset\\AlphaFold Protein Database e.coli\\Uncompressed'.
    output_dir : output folder directory, optional
        DESCRIPTION. The default is 'dataset\\AlphaFold Protein Database e.coli\\Uncompressed\\graphs'.
    threshold : TYPE, optional
        DESCRIPTION. The default is 6.0; threshold for molecular interaction.

    Returns
    -------
    None.

    """
    
    # Enumerates over all files in the directory with a .pdb extension.
    for file in os.listdir(direct):
        if '.pdb' in file:
            parse(fr'{direct}/{file}', output_dir, threshold=6.0);
    return;