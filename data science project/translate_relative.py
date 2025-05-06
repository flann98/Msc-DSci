# -*- coding: utf-8 -*-
"""
Created on Wed Apr 09 16:41:XX 2025

@author: brend
"""

import os;
import gzip;
import shutil;
# Requires BioPython installation.
from Bio.PDB import PDBParser;
import networkx as nx;
import numpy as np;

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
    Parses the individual .pdb file to create a graph for their amino acids.
    
    Parameters
    ----------
    file : .pdb file to be parsed.

    Returns
    -------
    AA_coordinates : Cartesian coordinates of the amino acid centroid.
    AA_types : Atomic types present in the protein.
    edges : Edges as neighbours to amino acids within 6.0 of one another.

    """
    
    # Fetching the unique ID for the file to save the graphs under.
    fileID, _ = os.path.splitext(os.path.basename(file));
    
    try:
        # Parse the PDB file.
        parser = PDBParser();
        # Fetching the structure of the protein.
        structure = parser.get_structure('protein', file);
        
        # Create the graph format.
        graph = nx.Graph();
        
        # Initialise amino acid labels.
        amino_acids = [];

        # Extract residues and atoms.
        for model in structure:
            for chain in model:
                # Looping to the residue level of the structure.
                for residue in chain:
                    # Initialise atomic coordinates list.
                    atom_coords = [];
                    coord0 = [];
                    coord1 = [];
                    coord2 = [];
                    # Append current residue of amino acid sequence.
                    amino_acids.append(residue.resname);
                    # Looping to the atomic layer of the structure.
                    for atom in residue:
                        # Appending atomic coordinates for centroiding.
                        atom_coords.append(atom.coord);
                    for k in range(len(atom_coords)):
                        coord0.append(atom_coords[k][0]);
                        coord1.append(atom_coords[k][1]);
                        coord2.append(atom_coords[k][2]);
                    res_node = residue.id[1];
                    graph.add_node(res_node, 
                                       chain=chain.id, 
                                       coordX=np.mean(coord0),
                                       coordY=np.mean(coord1),
                                       coordZ=np.mean(coord2));
                    # Add edges based on distance threshold.
                    add_edges = [];
                    for other_res_id, other_res in graph.nodes(data="residue"):
                        if other_res_id != res_node:
                            # Calculates euclidean distance between points.
                            dist = np.linalg.norm(np.array((graph.nodes[res_node]['coordX'],
                                                           graph.nodes[res_node]['coordY'],
                                                           graph.nodes[res_node]['coordZ'])) \
                                                  - np.array((graph.nodes[other_res_id]['coordX'],
                                                             graph.nodes[other_res_id]['coordY'],
                                                             graph.nodes[other_res_id]['coordZ'])));
                            # Threshold for interaction.
                            if abs(dist) <= threshold:
                                add_edges.append([res_node, other_res_id, dist]);
                    for new in add_edges:
                        graph.add_edge(new[0], new[1], weight=new[2]);
        # Altering the node coordinates at the end to be relative.
        graph_copy = graph.copy();
        for n in np.arange(1, len(graph.nodes) + 1, step=1):
            if n == 1:
                graph.nodes[n]['coordX'] = 0;
                graph.nodes[n]['coordY'] = 0;
                graph.nodes[n]['coordZ'] = 0;
            else:
                graph.nodes[n]['coordX'] = \
                    graph_copy.nodes[n]['coordX'] - \
                        graph_copy.nodes[n - 1]['coordX'];
                graph.nodes[n]['coordY'] = \
                    graph_copy.nodes[n]['coordY'] - \
                        graph_copy.nodes[n - 1]['coordY'];
                graph.nodes[n]['coordZ'] = \
                    graph_copy.nodes[n]['coordZ'] - \
                        graph_copy.nodes[n - 1]['coordZ'];
        
        # Save the amino acid sequence as labels.
        with open(fr'{output_dir}/{fileID}-AAlabel.csv', 
                  mode='w', 
                  newline='') as fmt:
            for acid in amino_acids:
                fmt.write(f'{acid}\n');
        # Save the graph as graphML format.
        nx.write_graphml(graph,
                         fr'{output_dir}/{fileID}-graph.graphml');
        return;

    except Exception as e:
        print(f"Error parsing PDB file: {e}");
        return None;

def iterator(direct=r'dataset/AlphaFold Protein Database e.coli/Uncompressed',
             output_dir=r'dataset/AlphaFold Protein Database e.coli/Uncompressed/graphs/graphML2', 
             threshold=6.0):
    """
    Iterates over the entire directory with the parse function.

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
            parse(fr'{direct}/{file}', output_dir, threshold=threshold);
    return;