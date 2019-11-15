import h5py
import torch
from torch_scatter import scatter_add

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

from check_err_deg import degree


##### some default parameters
botnet = 'leet'
graph_id = 60

lay = 6
eg = True

data_dir = '/n/holylfs/LABS/tata_yu_rush_level1_data/TataProject/data_botnet'
data_name = f'{botnet}_no100k_ev10k_udr_slp_test.hdf5'

model_dir = './saved_models'
model_name = f'{botnet}_no100k_ev10k_us_model_lay{lay}_rh1_eg_ep50.pt' if eg else \
             f'{botnet}_no100k_ev10k_us_model_lay{lay}_rh1_ep50.pt'

plt_save_dir = './plots'
plt_save_name = f'{botnet}_test_g{graph_id}_model_lay{lay}_eg_act' if eg else \
                f'{botnet}_test_g{graph_id}_model_lay{lay}_act'

##### some functions
def print_file_attrs(f):
    for k, v in f.attrs.items():
        print(k, v)
    return


if __name__ == '__main__':
    f = h5py.File(os.path.join(data_dir, data_name), 'r')
    x = torch.tensor(f[str(graph_id)]['x'][()]).cuda()
    edge_index = torch.tensor(f[str(graph_id)]['edge_index']).cuda()
    y = torch.tensor(f[str(graph_id)]['y'][()]).long().cuda()
    num_nodes = f[str(graph_id)].attrs['num_nodes']

    ##### node degree distribution
    deg = degree(edge_index, num_nodes)
    degf = deg.float()
    print(f'file name: {data_name} | graph id: {graph_id}')
    print_file_attrs(f)
    print('\n')
    print('node degree distribution:')
    print(f'average: {degf.mean().item()}, >100: {(deg > 100).sum().item()}, >1000: {(deg > 1000).sum().item()}')
    print('normal node degree distribution:')
    print(f'average: {degf[y==0].mean().item()}, >100: {(deg[y==0] > 100).sum().item()}, >1000: {(deg[y==0] > 1000).sum().item()}')
    print('botnet node degree distribution:')
    print(f'average: {degf[y==1].mean().item()}, >100: {(deg[y==1] > 100).sum().item()}, >1000: {(deg[y==1] > 1000).sum().item()}')
 
    ##### check the prediction and all the middle activations of a learned model
    print('\n')
    print(f'model name: {model_name}')
    model = torch.load(os.path.join(model_dir, model_name), map_location='cuda:0')
    with torch.no_grad():
        model.eval()
        xoo = model(x, edge_index)
    pred = xoo.argmax(dim=1)
    rec = recall(pred, y)
    fn = ((pred == 0) & (y == 1)).nonzero()
    fp = ((pred == 1) & (y == 0)).nonzero()
    print(f'number of false positives: {len(fp)}')
    print(f'number of false negatives: {len(fn)}')
    xl = [x]
    xr = [x]
    with torch.no_grad():
        model.eval()
        for i in range(lay):
            temp = model.gcn_net[i](xl[-1], edge_index)
            tempr = model.residuals[i](xl[-1])
            tx = model.non_linear(temp + tempr)
            xl.append(tx)
            xr.append(tempr)
        xo = model.final(xl[-1])
    xln = [a.norm(dim=1) for a in xl]
    xrn = [a.norm(dim=1) for a in xr]
#    breakpoint()
    ##### check the degrees, activation norms, of the fp and fn nodes
    plt.figure(figsize=(10, 8))
    for i in range(lay):
#        plt.plot(deg[fn].cpu().numpy(), xln[i][fn].cpu().numpy(), 'o-', label=f'layer {i+1}')
        plt.scatter(deg[fn].cpu().numpy(), xln[i][fn].cpu().numpy(), marker='o', label=f'layer {i + 1}')
    plt.legend()
    plt.xlabel('node degrees')
    plt.ylabel('layer activation magnitudes')
    plt.title('flase negative nodes')

    plt.savefig(os.path.join(plt_save_dir, plt_save_name))
    print('\n')
    print(f'plot saved with name: {plt_save_name}')

