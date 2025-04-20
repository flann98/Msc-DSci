# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:40:29 2025

@author: brend
"""

import os;
import csv;
from collections import Counter;
import numpy as np;
from scipy.stats import mode as SPmode;
import matplotlib.pyplot as plt;

def AAlabel_read(path=r'dataset/AlphaFold Protein Database e.coli/Uncompressed/graphs/graphML'):
    """
    Reads all proteins' amino acid labels to create a dict which contains them.
    
    Parameters
    ----------
    path : String as a path to the file, optional
    -Default: r'dataset/AlphaFold Protein Database e.coli/Uncompressed/graphs'.

    Returns
    -------
    labels : Dict containing the proteins' associated amino acid labels.

    """
    
    labels = dict();
    
    # Iterates over ever labelling amino acid sequence .csv to add to the dict.
    for file in os.listdir(path):
        if '.csv' in file:
            with open(rf'{path}/{file}', mode='r') as open_file:
                # Creating the value of amino acid sequence for each protein dict.
                labels[rf'{file}'] = [acid for row in csv.reader(open_file) for acid in row];
    
    return labels;

def class_balance(labels):
    """
    Generates figures to illustrate potential class imbalances within the data.

    Parameters
    ----------
    labels : The amino acid sequence dict which contains the composition of all
    proteins within the proteome dataset.

    Returns
    -------
    None.

    """
    
    # Flattens the amino acids into a long list.
    label_list = [acid for protein in labels.values() for acid in protein];
    label_sum = Counter(label_list);
    # Creating arrays with the value counts and the counts' labels.
    sum_vals = np.array(list(label_sum.values()), dtype='int32');
    sum_type = np.array(list(label_sum.keys()), dtype='str');
    
    fig, ax = plt.subplots(dpi=144);
    ax.pie(sum_vals, labels=sum_type, autopct='%2.1f%%', pctdistance=1.1,
           labeldistance = 0.7, textprops=dict(horizontalalignment='center',
                                               verticalalignment='bottom', 
                                               size=6));
    ax.set_title('The proportional representation of amino acids in the proteome.',
                 fontsize=8);
    
    return;
    
def sequence_length(labels):
    """
    Generates figures to illustrate length imbalances between proteins.

    Parameters
    ----------
    labels : The amino acid sequence dict which contains the composition of all
    proteins within the proteome dataset.

    Returns
    -------
    None.

    """
    
    # Creating an array of all the counts of sequence length.
    sequence_lengths = [];
    for s in labels.values():
        sequence_lengths.append(len(s));
    # Creating matching variables for the number of occurences and the seq length.
    mode = SPmode(sequence_lengths);
    quantity = np.unique(sequence_lengths, return_counts=True);
    
    # Plotting the proportion of sequence lengths.
    fig, ax = plt.subplots(dpi=144);
    ax.bar(quantity[0], quantity[1]);
    ax.set_title('Length of amino acid sequence for proteins' \
                 ' plotted against occurences in the proteome.', 
                 fontsize=8);
    ax.set_xlabel('Sequence length', fontsize=8);
    ax.set_ylabel('Instances of sequence lengths', fontsize=8);
    ax.set_ylim(0, np.max(quantity[1]));
    ax.set_xlim(0, np.sort(quantity[0])[-10]);
    ax.set_yticks(np.arange(0, np.max(quantity[1]) + 1, step=1), minor=True);
    ax.set_yticks(np.arange(0, np.max(quantity[1]) + 1, step=5), minor=False);
    ax.set_xticks(np.arange(0, np.sort(quantity[0])[-10], step=100));
    ax.tick_params(axis='x', labelsize=6, labelrotation=45);
    ax.tick_params(axis='y', labelsize=6, labelrotation=15);
    ax.grid(visible=True, which='major', axis='y', linestyle=':', linewidth=0.4);
    ax.text(np.sort(quantity[0])[-10] - 50, np.max(quantity[1]) - 1, 
            s=f'10 longest sequences:\n{np.sort(quantity[0])[-10:]}' \
                f'\n\n Modal sequence length: {mode[0]}' \
                    f'\n Modal count: {mode[1]}' \
                        f'\n Mean sequence length: {np.mean(quantity[0])}',
                        ha='right', va='top', fontsize=8);
    # Blown up plot from a limited range.
    fig, ax = plt.subplots(dpi=144);
    ax.set_title('Length of amino acid sequence for proteins' \
                 '\nplotted against occurences in the proteome.', fontsize=10);
    ax.set_xlabel('Sequence length', fontsize=8);
    ax.set_ylabel('Instances of sequence lengths', fontsize=8);
    ax.bar(quantity[0], quantity[1]);
    ax.set_ylim(0, np.max(quantity[1]));
    ax.set_xlim(0, 600);
    ax.set_yticks(np.arange(0, np.max(quantity[1]) + 1, step=1));
    ax.set_xticks(np.arange(0, 601, step=25));
    ax.tick_params(axis='x', labelsize=6, labelrotation=45);
    ax.tick_params(axis='y', labelsize=6, labelrotation=15);
    ax.grid(visible=True, axis='y', linestyle=':', linewidth=0.4);
    
    
labels = AAlabel_read();
class_balance(labels);
sequence_length(labels);