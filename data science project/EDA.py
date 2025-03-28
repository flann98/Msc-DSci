# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:40:29 2025

@author: brend
"""



import os;
import csv;
from collections import Counter;
import matplotlib.pyplot as plt;
import pandas as pd;
import numpy as np;
from sklearn import model_selection;
# import tensorflow as tf;
# Requires pip install of tensorflow_gnn.
#import tensorflow-gnn as tfgnn;

def AAlabel_read(path=r'dataset/AlphaFold Protein Database e.coli/Uncompressed/graphs'):
    """
    Reads all proteins' amino acid labels to create a dict which contains them.
    
    Parameters
    ----------
    path : String as a path to the file, optional
        DESCRIPTION. Default: r'dataset/AlphaFold Protein Database e.coli/Uncompressed/graphs'.

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
    ax.pie(sum_vals, labels=sum_type, autopct='%1.1f%%');
    
    return;