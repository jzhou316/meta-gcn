'''
Given a graph dataset stored in a HDF5 file, count the degrees of every node in each graph,
and store the degrees in node features "x" (apended).

Note: default is the out-degree (for both directed and undirected graphs).
'''

import os
import argparse

import h5py
from tqdm import tqdm

import torch
from torch_scatter import scatter_add
from torch_geometric.utils.num_nodes import maybe_num_nodes


##### some default parameters
data_dir = '/n/holylfs/LABS/tata_yu_rush_level1_data/xuzhiying/public_data_processed'
'''
data_file = 'chord/chord_no100k_ev10k_us.hdf5'
data_file = 'debru/debru_no100k_ev10k_us.hdf5'
data_file = 'kadem/kadem_no100k_ev10k_us.hdf5'
data_file = 'leet/leet_no100k_ev10k_us.hdf5'
'''
degree_type = 'out'
append_to_x = True
#####


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate node degrees and store in file',
                                     prog='add_degrees',
                                     epilog='example: python data_add_degree.py chord/chord_no100k_ev10k_us.hdf5')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='data directory')
    parser.add_argument('data_file', type=str, help='data file name (may include subpaths before the name)')
    parser.add_argument('--degree_type', type=str, choices=['out', 'in'], default=degree_type, help='out-degree (default) or in-degree')
    parser.add_argument('--append2x', type=int, default=append_to_x, help='whether to append node degrees to the node feature "x"')
    args = parser.parse_args()
    return args


##### some functions
def degree(edge_index, num_nodes, degree_type='out'):
    '''
    Calculate node degrees in a graph.

    Input:
        edge_index (torch.LongTensor): size (2, num_edges)
        num_nodes (int): number of nodes
        degree_type (str, optional): choose from ['out', 'in']. Default: 'out'.
                                     For an undirected graph, this doesn't make a difference.

    Output:
        deg (torch.Tensor): size (num_nodes,), node degrees. 
    '''
    assert degree_type in ['out', 'in']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) if degree_type == 'out' else \
          scatter_add(edge_weight, col, dim=0, dim_size=num_nodes) 

    return deg


def add_degree_file(path, degree_type='out', append_to_x=True):
    '''
    Calculate node degrees for all graphs stored in a HDF5 file, and store the degrees.

    Input:
        append_to_x (bool, optional): whether to append the degrees to node features 'x'.
                Default: True. If not, store the degrees in a new graph attribute with name
                'deg' (when degree_type is 'out' or 'indeg' when degree_type is 'in').

    Note:
        - Default is the out-degree.
        - In the input file, each graph must have attribute 'num_nodes'.
    '''
    assert degree_type in ['out', 'in']
    with h5py.File(path, 'r+') as f:
        for graph_id in tqdm(f):
            deg = degree(torch.tensor(f[graph_id]['edge_index'][()]), f[graph_id].attrs['num_nodes'], degree_type)
            if append_to_x:
                x = torch.cat([torch.tensor(f[graph_id]['x'][()]), deg.view(-1, 1)], dim=1)
                del f[graph_id]['x']
                f[graph_id].create_dataset('x', shape=tuple(x.size()), dtype='f4', data=x)
            else:
                if degree_type == 'out':
                    f[graph_id].create_dataset('deg', shape=tuple(deg.size()), dtype='f4',
                                               data=deg)
                else:
                    f[graph_id].create_dataset('indeg', shape=tuple(deg.size()), dtype='f4',
                                               data=deg)
    print(f'Graph data augmented in {path}')
    print(f'Size: {os.path.getsize(path) / 1000**2} MB | {os.path.getsize(path) / 1000**3} GB')

    return None


if __name__ == '__main__':
    args = parse_args()

    data_path = os.path.join(args.data_dir, args.data_file)

    add_degree_file(data_path, degree_type=args.degree_type, append_to_x=args.append2x)

