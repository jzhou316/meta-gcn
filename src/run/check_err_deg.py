import h5py
import torch
from torch_scatter import scatter_add

from collections import Counter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
from metrics import *
import sys
sys.path.insert(0, '../data')
from dataset import DSTGraphDataset
from dataloader import DSTGraphLoader
sys.path.insert(0, '../models')
from dstgcn_model import DstGCNModel


##### some default parameters
botnet = 'leet'
graph_id = 10

data_dir = '/n/holylfs/LABS/tata_yu_rush_level1_data/TataProject/data_botnet'
data_name = f'{botnet}_no100k_ev10k_udr_slp_test.hdf5'

model_dir = './saved_models'
model_name = f'{botnet}_no100k_ev10k_us_model_lay10_rh1_ep50.pt'

plt_save_dir = './plots'
plt_save_name = f'{botnet}_test_g{graph_id}'

##### some functions
def degree(edge_index, num_nodes):
    '''
    Calculate node degrees in a graph. Input is a undirected graph.
    '''
    edge_weight = edge_index.new_ones((edge_index.size(1), ))
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    return deg

def neighbors(edge_index, node_id):
    '''
    Return a node's neighbors. Input is a undirected graph.
    '''
    row, col = edge_index
    return col[row == node_id]


if __name__ == '__main__':
    f = h5py.File(os.path.join(data_dir, data_name), 'r')
    x = torch.tensor(f[str(graph_id)]['x'][()]).cuda()
    edge_index = torch.tensor(f[str(graph_id)]['edge_index']).cuda()
    y = torch.tensor(f[str(graph_id)]['y'][()]).long().cuda()
    num_nodes = f[str(graph_id)].attrs['num_nodes']

    num_bins = 10
    ##### node degree distribution
    deg = degree(edge_index, num_nodes)
    plt.figure()
    plt.suptitle('node degree distribution')
    plt.subplot(311)
    plt.hist(deg.cpu(), num_bins)
#    c = Counter(deg.cpu())
#    plt.bar(c.keys(), c.values())

    plt.subplot(312)
    plt.hist(deg[y == 1].cpu(), num_bins)
#    c = Counter(deg[y == 1].cpu())
#    plt.bar(c.keys(), c.values())
    plt.title('evils (positive) nodes')
    
    plt.subplot(313)
    plt.hist(deg[y == 0].cpu(), num_bins)
#    c = Counter(deg[y == 0].cpu())
#    plt.bar(c.keys(), c.values())
    plt.title('normal (negative) nodes')
    
    plt.savefig(os.path.join(plt_save_dir, plt_save_name))
    
    ##### check the prediction of a learned model
    model = torch.load(os.path.join(model_dir, model_name), map_location='cuda:0')
    with torch.no_grad():
        model.eval()
        xo = model(x, edge_index)
    pred = xo.argmax(dim=1)
    rec = recall(pred, y)
    fp_nodes = ((pred == 1) & (y == 0)).nonzero()
    fn_nodes = ((pred == 0) & (y == 1)).nonzero()
    
    fp_nodes_deg = deg[fp_nodes]
    fn_nodes_deg = deg[fn_nodes]

    plt.figure()
    plt.subplot(211)
    plt.hist(fp_nodes_deg.cpu(), num_bins)
#    c = Counter(fp_nodes_deg.cpu())
#    plt.bar(c.keys(), c.values())
    plt.title(f'false positive node degrees (num: {len(fp_nodes)})')
    plt.subplot(212)
    plt.hist(fn_nodes_deg.cpu(), num_bins)
#    c = Counter(fn_nodes_deg.cpu())
#    plt.bar(c.keys(), c.values())
    plt.title(f'false negative node degrees (num: {len(fn_nodes)})')
    plt.savefig(os.path.join(plt_save_dir, plt_save_name) + '_err')

    ##### check the neighbors of the error nodes
    fp_nodes_neibor1 = [(y[neighbors(edge_index, id)] == 1).sum().item() for id in fp_nodes]
    fn_nodes_neibor1 = [(y[neighbors(edge_index, id)] == 1).sum().item() for id in fn_nodes]
    fp_nodes_neibor1_frac = [a/b for a, b in zip(fp_nodes_neibor1, fp_nodes_deg.cpu().float())]
    fn_nodes_neibor1_frac = [a/b for a, b in zip(fn_nodes_neibor1, fn_nodes_deg.cpu().float())]

    plt.figure()
    plt.subplot(221)
#    plt.hist(fp_nodes_neibor1, num_bins)
    c = Counter(fp_nodes_neibor1)
    plt.bar(c.keys(), c.values())
    plt.title(f'fp node nbrs being evil (num: {len(fp_nodes)})')
    plt.subplot(222)
#    plt.hist(fn_nodes_neibor1, num_bins)
    c = Counter(fn_nodes_neibor1)
    plt.bar(c.keys(), c.values())
    plt.title(f'fn node nbrs being evil (num: {len(fn_nodes)})')
    plt.subplot(223)
    plt.hist(fp_nodes_neibor1_frac, num_bins)
    plt.title('fp node nbrs being evil fraction')
    plt.subplot(224)
    plt.hist(fn_nodes_neibor1_frac, num_bins)
    plt.title('fn node nbrs being evil fraction')
    plt.savefig(os.path.join(plt_save_dir, plt_save_name) + '_err_nbr')
    
    


