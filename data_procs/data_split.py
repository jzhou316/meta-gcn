'''
Split the dataset into train/val/test sets.
'''

import os
import random
import math
import argparse

import h5py
from tqdm import tqdm


##### some default arguments
data_dir = '/n/holylfs/LABS/tata_yu_rush_level1_data/xuzhiying/public_data_processed'
'''
data_file = 'chord/chord_no100k_ev10k_us.hdf5'
data_file = 'debru/debru_no100k_ev10k_us.hdf5'
data_file = 'kadem/kadem_no100k_ev10k_us.hdf5'
data_file = 'leet/leet_no100k_ev10k_us.hdf5'
'''
save_dir = '../data/botnet/processed'
seed = 0
split_ratio = [8, 1, 1]
#####


def parse_args():
    parser = argparse.ArgumentParser(description='Split a dataset into train/val/test sets \
                                                  (for static graphs stroed in one HDF5 file)',
                                     epilog='example: python data_split.py chord/chord_no100k_ev10k_us.hdf5')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='data directory')
    parser.add_argument('data_file', type=str, help='data file name (may include subpaths before the name)')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='new data directory')
    parser.add_argument('--seed', type=int, default=seed, help='random seed')
    parser.add_argument('--split_ratio', type=int, nargs='*', default=split_ratio, help='split ratio (2 values for train/test or 3 values for train/val/test)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(seed)
    data_file_path = os.path.join(args.data_dir, args.data_file)

    # create the names of the new files
    data_file_name, data_file_ext = os.path.splitext(os.path.basename(args.data_file))
    if len(split_ratio) == 3:
        split_labels = ['train', 'val', 'test']
    elif len(split_ratio) == 2:
        split_labels = ['train', 'test']
    else:
        raise ValueError('length of split ratio should be 2 or 3')

    for label in split_labels:
        vars()[f'{label}_file_name'] = os.path.join(args.save_dir, f'{data_file_name}_{label}{data_file_ext}')
        temp = vars()[f'{label}_file_name']
        if os.path.exists(temp):
            raise OSError(f'file exists: {temp}')

    # random permutation of the data
    with h5py.File(data_file_path, 'r') as f:
        num_total = f.attrs['num_graphs']
        srt = 0
        split_ranges = []
        for i, r in enumerate(split_ratio):
            if i < len(split_ratio) - 1:
                num = math.floor(r / sum(split_ratio) * num_total)
                split_ranges.append(range(srt, srt + num))
                srt = srt + num
            else:
                split_ranges.append(range(srt, num_total))
        
        graph_ids = list(range(num_total))
        random.shuffle(graph_ids)

        for label, r in zip(split_labels, split_ranges):
            print(f'creating {label} set ' + '-' * 10)
            with h5py.File(vars()[f'{label}_file_name'], 'w') as g:
                num_nodes_sum = 0
                num_edges_sum = 0
                num_evils_sum = 0
                if 'num_evil_edges_avg' in f.attrs:
                    num_evil_edges_sum = 0
                    num_evil_edges_flag = True
                else:
                    num_evil_edges_sum = None
                    num_evil_edges_flag = False
                ori_graph_ids = []
                for n, i in tqdm(enumerate(r)):
                    f.copy(str(graph_ids[i]), g, name=str(n))
                    num_nodes_sum += f[str(graph_ids[i])].attrs['num_nodes']
                    num_edges_sum += f[str(graph_ids[i])].attrs['num_edges']
                    num_evils_sum += f[str(graph_ids[i])].attrs['num_evils']
                    if num_evil_edges_flag:
                        try:
                            num_evil_edges_sum += f[str(graph_ids[i])].attrs['num_evil_edges']
                        except:
                            num_evil_edges_flag = False
                    ori_graph_ids.append(graph_ids[i])
                g.attrs['num_graphs'] = n + 1
                g.attrs['num_nodes_avg'] = num_nodes_sum / (n + 1)
                g.attrs['num_edges_avg'] = num_edges_sum / (n + 1)
                g.attrs['num_evils_avg'] = num_evils_sum / (n + 1)
                if num_evil_edges_flag:
                    g.attrs['num_evil_edges_avg'] = num_evil_edges_sum / (n + 1)
                g.attrs['is_directed'] = f.attrs['is_directed']
                g.attrs['contains_self_loops'] = f.attrs['contains_self_loops']
                g.attrs['ori_graph_ids'] = ori_graph_ids
            print('number of graphs: {}, data saved at {}.'.format(n + 1, vars()[f'{label}_file_name']))


