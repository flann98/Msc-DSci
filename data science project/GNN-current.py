# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 13:35:58 2025

@author: brend
"""

import os;
import math;
import random;
import networkx as nx;
import pandas as pd;
import torch;
from torch.nn import Linear;
from torch.nn.functional import relu;
from torch_geometric.nn import MessagePassing, GATConv;
from torch_geometric.data import Data;
from torch_geometric.loader import DataLoader;


def load_amino_acid_labels(csv_file, amino_acids):
    """
    Loads amino acid labels from a CSV file.

    Parameters
    ----------
    csv_file : The csv file in the directory to be processed.
    amino_acids : The constant 

    Returns
    -------
    Returns the one-hot encoded amino acid features (sequence).

    """
    
    seq = pd.read_csv(csv_file, header=None);
    # One-hot encoding for amino acid features.
    seq = seq[0].to_numpy();
    aa_to_OH = [];
    for i, j in enumerate(seq):
        aa_to_OH.append(amino_acids[j]);
    aa_tensor = torch.tensor(aa_to_OH, dtype=torch.int);
    return aa_tensor;

def load_graph(graph_file):
    """
    Loads the molecular graph of the protein from the graph file.

    Parameters
    ----------
    graph_file: The graph file in the directory to be processed.

    Returns
    -------
    Returns the networkx graph containing the nodes and their related edges.

    """
    # Utilises the networkx read_graphml function to extract nodes & edges.
    graph = nx.read_graphml(graph_file);
    node_data = [];
    # Necessary to remove strings from the tensor data.
    for node, coord in graph.nodes(data=True):
        node_index = int(node);
        coords = [coord['coordX'], coord['coordY'], coord['coordZ']];
        # Combines the integer node with its corresponding coordinate floats.
        node_data.append([node_index] + coords);
    # Resulting 32 bit float tensor:
    node_tensor = torch.tensor(node_data, dtype=torch.float);
    edge_data = [];
    edge_weight = [];
    # Similar operation for the edge list.
    for source, target, weight in graph.edges(data=True):
        # Combines the edge with its corresponding weighting.
        edge_data.append([int(source) - 1, int(target) - 1]);
        edge_weight.append(float(weight['weight']));
        # Making the graph undirected:
        edge_data.append([int(target) - 1, int(source) - 1]);
        edge_weight.append(float(weight['weight']));
    # Resulting 32 bit float tensor:
    edge_tensor = torch.tensor(edge_data, dtype=torch.long);
    # Creating weight tensor for edge attribution.
    weight_tensor = torch.tensor(edge_weight, dtype=torch.float);
    
    return node_tensor, edge_tensor, weight_tensor;

def integrate_labels_and_graphs(directory=r'dataset/AlphaFold Protein Database e.coli/Uncompressed/graphs/graphML'):
    """
    Combines CSV and graphs into PyTorch Geometric Data objects across 
    directory by calling load_amino_acid_labels and load_graph6/sparse6/graphml.

    Parameters
    ----------
    directory : The path to the directory containing the file pairings.
    -Default: r'dataset/AlphaFold Protein Database e.coli/Uncompressed/graphs/graphML'.

    Returns
    -------
    data_list : The geometric data of one-hot encoded amino acid sequences and 
    atomic node.
    amino_acids : The list of amino acids 

    """
    
    # Creating a dict for one-hot encoding amino acids to a numeric value.    
    amino_acids = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
        'SER', 'THR', 'TRP', 'TYR', 'VAL', 
        'PYL', 'SEC'  
        ];
    AA_onehot = {aa : i for i, aa in enumerate(set(amino_acids))};
    
    # Initialising array.
    data_list = [];
    # Iterates over the entire directory of file-pairings.
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith('.csv'):
            # Taking the file name without the extension.
            base_name = file_name.split('AAlabel.csv')[0];
            csv_file = os.path.join(directory, file_name)
            graph_file = os.path.join(directory, f'{base_name}gcn_graph.graphml');
            # Checking the graph file path's validity.
            if os.path.exists(graph_file):
                AA_labels = load_amino_acid_labels(csv_file, AA_onehot);
                node_list, edge_list, weight_list = load_graph(graph_file);
                data = Data(x=node_list, 
                            edge_index=edge_list.t().contiguous(),
                            edge_attr=weight_list.contiguous(),
                            y=AA_labels);
                data_list.append(data);
                if len(data_list) % 10 == 0:
                    print(f'Processed {len(data_list)} entries');
    return data_list;

class GNN(MessagePassing):
    def __init__(self):
        super().__init__();
        self.conv1=GATConv(in_channels=-1, 
                           out_channels=4096);
        self.conv2=GATConv(in_channels=4096, 
                           out_channels=2048);
        self.fc=Linear(in_features=2048,
                        out_features=4);
        
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr;
        x = self.conv1(x, edge_index, edge_weight);
        x = relu(x);
        x = self.conv2(x, edge_index, edge_weight);
        x = relu(x);
        return self.fc(x);
                    
def test_model(loader, model, epoch, epochs,
               test=False, criterion=torch.nn.MSELoss()):
    
    model.eval();
    total_loss=0;
    for data in loader:
        if test:
            data_test = data.copy();
            data_test.x * 0;
            node_out = model(data_test);
            total_loss += criterion(node_out, data.x);
        else:
            data_val = data.copy();
            if epochs - epoch < 12:
                rank = epoch - (epochs - 12);
                data_val.x[math.ceil(rank/4)] * 0;
                if rank > 4:
                    data_val.x[math.ceil(rank/4) - (math.ceil(rank-4/4) * 2)];
            node_out = model(data_val);
            total_loss += criterion(node_out, data.x);
                

def train_model(loader, model, criterion=torch.nn.MSELoss(),
            chosen_opt=torch.optim.Adam, learn_rate=0.02):
    optimizer=chosen_opt(model.parameters(), lr=learn_rate);
    
    model.train();
    total_loss=0;
    for data in loader:
        optimizer.zero_grad();
        node_out = model(data);
        loss = criterion(node_out, data.x);
        loss.backward();
        optimizer.step();
        print(f'Loss: {loss.item():.4f}');
        total_loss += loss.item();
    return total_loss / len(loader);

def process(loader, model, epochs=16):
    
    for epoch in range(epochs):
        avg_loss = 0;
        avg_loss = train_model(loader[0], model);
        val_mse = test_model(loader[1], model, epoch);
        test_mse = test_model(loader[2], model, epoch, True);
        print(f'Epoch {epoch+1}, '\
              f'\nAverage MSE loss: {avg_loss:.4f}');
        

    

data_list = integrate_labels_and_graphs();
print(f'Processed {len(data_list)} file pairings into PyTorch Geometric Data objects.');
# Shuffling the Data objects' order in the list.
random.shuffle(data_list);
# Creating parameters for train/validation/test split.
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1;
train_size = int(train_ratio * len(data_list));
val_size = int(val_ratio * len(data_list));
# Creating the split lists.
train_data = data_list[:train_size];
# Sliced from the end of the train set to itself + the validation set.
val_data = data_list[train_size:train_size + val_size];
# Sliced from the end of the validation set to the end of full set.
test_data = data_list[train_size + val_size:];
# Creating the loaders for each split.
train_loader = DataLoader(train_data, batch_size=256);
val_loader = DataLoader(val_data, batch_size=256);
test_loader = DataLoader(test_data, batch_size=256);
# Training loop for the appropriate model.
process(loaders=[train_loader, val_loader, test_loader], model=GNN());
