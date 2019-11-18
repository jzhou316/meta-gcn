"""
Given a graph dataset stored in a HDF5 file, with each graph only contains directed edges,
make the graphs undirected by doubling the edges.

Also add self loops.

The graph here contains 'edge_y', which is the edge labels.
"""
import os
import argparse

from tqdm import tqdm
import h5py
import numpy as np
import torch

from undirected import to_undirected_ey
from loop import add_self_loops_ey


##### some default arguments
data_dir = '/n/holylfs/LABS/tata_yu_rush_level1_data/xuzhiying/public_data_processed'
'''
data_file = 'chord/chord_no100k_ev10k.hdf5'
data_file = 'debru/debru_no100k_ev10k.hdf5'
data_file = 'kadem/kadem_no100k_ev10k.hdf5'
data_file = 'leet/leet_no100k_ev10k.hdf5'
'''
save_dir = data_dir

if_to_undirected = True
if_add_self_loops = True

rewrite = False


#####

def parse_args():
    parser = argparse.ArgumentParser(description='Add undirected edges and self loops on directed graphs \
                                                  (stored in HDF5 files)',
                                     prog='add_edges',
                                     epilog='example: python data_add_edges.py chord/chord_no100k_ev10k.hdf5')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='data directory')
    parser.add_argument('data_file', type=str, help='data file name (may include subpaths before the name)')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='new data directory')
    parser.add_argument('--undirected', type=int, default=if_to_undirected, help='if to undirected')
    parser.add_argument('--self_loops', type=int, default=if_add_self_loops, help='if add self loops')
    parser.add_argument('--rewrite', type=int, default=rewrite, help='if rewrite the dataset, when both the \
                        "undirected" and "self_loops" flags are 0. The new file has "_new" appended to the \
                        original name.')

    args = parser.parse_args()
    return args


def extend_edges_ey(edge_index, edge_y=None, node_y=None, num_nodes=None,
                    if_to_undirected=True, if_add_self_loops=False):
    """
    Extend the edge indexes, which are directed and contain no self loops.
    Here the 'edge_y' binary labels are extended as well.
    For self-loops, their labels are according to the node labels.

    Note:
        - The input `edge_index` should not contain any undirected edges and self loops as we do not check them.
          However, even if undirected edges exist, the code will work fine by automatically removing duplicated
          edges and sort them. For the case of existing self-loops, we only add the remaining self-loops, to ensure
          no duplication.
        - The undirected edges are sorted, and self loop edges are appended at the end.
    """
    if isinstance(edge_index, np.ndarray):
        is_numpy = True
        edge_index = torch.tensor(edge_index)
        edge_y = torch.tensor(edge_y).byte() if edge_y is not None else None
        node_y = torch.tensor(node_y).byte() if node_y is not None else None
    elif isinstance(edge_index, torch.Tensor):
        is_numpy = False
    else:
        raise ValueError('input edge_index must be numpy.ndarray or torch.Tensor')

    if if_to_undirected:
        edge_index, edge_y = to_undirected_ey(edge_index, edge_y=edge_y, num_nodes=num_nodes)
    if if_add_self_loops:
        edge_index, edge_y = add_self_loops_ey(edge_index, edge_y=edge_y, node_y=node_y, num_nodes=num_nodes)

    if edge_y is not None:
        assert len(edge_y) == edge_index.size(1)

    return edge_index.numpy() if is_numpy else edge_index, edge_y.numpy() if is_numpy else edge_y


def extend_edges_file(path, savepath, if_to_undirected=False, if_add_self_loops=False, rewrite=False):
    """
    Extend the edge indexes for all graphs stored in a HDF5 file.

    Note:
        - In the input file, each graph must have attributes 'num_nodes' and 'num_edges'.
    """
    if not if_to_undirected and not if_add_self_loops:
        if not rewrite:
            print('No operation needed.')
            return None
        else:
            savepath = f'{os.path.splitext(savepath)[0]}_new{os.path.splitext(savepath)[1]}' \
                if savepath == path else savepath
            print(f'Rewrite {path} to {savepath}.')

    with h5py.File(path, 'r') as f, h5py.File(savepath, 'w') as g:
        num_graphs = 0
        num_edges_sum = 0
        num_evil_edges_sum = 0
        for n, graph_id in tqdm(enumerate(f)):
            edge_index, edge_y = extend_edges_ey(f[graph_id]['edge_index'][()],
                                                 edge_y=f[graph_id]['edge_y'][()] if 'edge_y' in f[graph_id] else None,
                                                 node_y=f[graph_id]['y'][()],
                                                 num_nodes=f[graph_id].attrs['num_nodes'],
                                                 if_to_undirected=if_to_undirected,
                                                 if_add_self_loops=if_add_self_loops)
            grp = g.create_group(graph_id)
            for p in f[graph_id]:
                if p == 'x':
                    grp.create_dataset(p,
                                       shape=f[graph_id][p].shape,
                                       dtype='f4',
                                       data=f[graph_id][p][()])
                elif p == 'y':
                    grp.create_dataset(p,
                                       shape=(f[graph_id][p].shape[0],),
                                       dtype='u1',
                                       data=f[graph_id][p][()])

                elif p == 'edge_index':
                    grp.create_dataset(p,
                                       dtype='i8',
                                       data=edge_index)
                elif p == 'edge_y':
                    grp.create_dataset(p,
                                       dtype='u1',
                                       data=edge_y)
                else:
                    raise ValueError('We currently only have 4 types of data stored for each graph.')

            for k, i in f[graph_id].attrs.items():
                if k == 'num_edges':
                    grp.attrs[k] = i + (i if if_to_undirected else 0) + \
                                   (f[graph_id].attrs['num_nodes'] if if_add_self_loops else 0)
                    num_edges_sum += grp.attrs[k]
                if k == 'num_evil_edges':
                    grp.attrs[k] = grp['edge_y'][()].sum()        # this only works for binary edge labels
                    num_evil_edges_sum += grp.attrs[k]
                elif k == 'is_directed':
                    grp.attrs[k] = 0 if if_to_undirected else i
                elif k == 'contains_self_loops':
                    grp.attrs[k] = 1 if if_add_self_loops else i
                else:
                    grp.attrs[k] = i

        num_graphs = n + 1
        for k, i in f.attrs.items():
            if k == 'num_edges_avg':
                g.attrs[k] = num_edges_sum / num_graphs
            elif k == 'num_evil_edges_avg':
                g.attrs[k] = num_evil_edges_sum / num_graphs
            elif k == 'is_directed':
                g.attrs[k] = 0 if if_to_undirected else i
            elif k == 'contains_self_loops':
                g.attrs[k] = 1 if if_add_self_loops else i
            else:
                g.attrs[k] = i

    print(f'New graph data saved in {savepath}')
    print(f'Size: {os.path.getsize(savepath) / 1000 ** 2} MB | {os.path.getsize(savepath) / 1000 ** 3} GB')

    return None


if __name__ == '__main__':
    args = parse_args()

    data_path = os.path.join(args.data_dir, args.data_file)

    save_data_name = os.path.splitext(args.data_file)[0] \
                     + '_' \
                     + ('u' if args.undirected else str()) \
                     + ('s' if args.self_loops else str()) \
                     + ('new' if not args.undirected and not args.self_loops else str()) \
                     + os.path.splitext(args.data_file)[1]
    save_data_path = os.path.join(args.save_dir, save_data_name)

    extend_edges_file(data_path, save_data_path, if_to_undirected, if_add_self_loops, rewrite)
