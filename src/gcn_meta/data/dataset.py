import h5py
import deepdish as dd
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T

from .data import GraphData
from .utils import h5group_to_dict


class GraphDataset(Dataset):
    """
    Graph dataset for a list of static graphs.
    The graphs are stored in a HDF5 file.

    Args:
        path (str): path of the HDF5 file containing a series of graph data
        num_graphs (int, optional): number of graphs in the dataset
        in_memory (bool, optional): whether to read all the graphs into memory. Default: False
    """
    def __init__(self, path, num_graphs=None, in_memory=False):
        super().__init__()

        self.path = path
        self.num_graphs = num_graphs
        self.in_memory = in_memory

        if in_memory:
            self.data = dd.io.load(path)  # dictionary
            self.data_type = 'dict'
            self.num_graphs = self.data['num_graphs']
        else:
            self.data = h5py.File(path, 'r')
            self.data_type = 'file'
            self.num_graphs = self.data.attrs['num_graphs']

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, index):
        if self.data_type == 'dict':
            return GraphData(self.data[str(index)])
        elif self.data_type == 'file':
            return GraphData(h5group_to_dict(self.data[str(index)]))
        else:
            raise ValueError


class NormalizedDegree(object):
    """
    Taken from https://github.com/anonymousOPT/OTCoarsening/blob/master/src/datasets.py.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class NodeFeatureOnes:
    """
    Put ones as node features for featureless graph.
    """
    def __init__(self):
        pass

    def __call__(self, data):
        data.x = torch.ones(data.num_nodes, 1)
        return data


class MyTUDataset(TUDataset):
    """
    Graph classification dataset. Adapted from TUDataset, and
    https://github.com/anonymousOPT/OTCoarsening/blob/master/src/datasets.py.
    """
    def __init__(self, path, name, x_deg=True, add_sl=False):
        super().__init__(path, name)

        if add_sl:
            '''this doesn't work, as the data are copied when indexed out
            # for data in self:
            #     data.edge_index, _ = add_remaining_self_loops(data.edge_index)
            '''
            data_list = []
            for data in self:
                data.edge_index, _ = add_remaining_self_loops(data.edge_index)
                data_list.append(data)
            self.data, self.slices = self.collate(data_list)
            # refer to https://github.com/rusty1s/pytorch_geometric/blob/eacb7d3e24a28aa50d7ae6d20a42676bc8ca1536
            # /torch_geometric/datasets/tu_dataset.py#L140

        if self.data.x is None:
            if x_deg:
                max_degree = 0
                degs = []
                for data in self:
                    degs += [degree(data.edge_index[0], dtype=torch.long)]
                    max_degree = max(max_degree, degs[-1].max().item())

                if max_degree < 1000:
                    self.transform = T.OneHotDegree(max_degree)
                else:
                    deg = torch.cat(degs, dim=0).to(torch.float)
                    mean, std = deg.mean().item(), deg.std().item()
                    self.transform = NormalizedDegree(mean, std)
            else:
                self.transform = NodeFeatureOnes()


class MyPPI(PPI):
    """
    PPI dataset for node classification.
    """
    def __init__(self, root, split, add_sl=False):
        super().__init__(root, split)

        if add_sl:
            '''this doesn't work, as the data are copied when indexed out
            # for data in self:
            #     data.edge_index, _ = add_remaining_self_loops(data.edge_index)
            '''
            data_list = []
            for data in self:
                data.edge_index, _ = add_remaining_self_loops(data.edge_index)
                data_list.append(data)
            self.data, self.slices = self.collate(data_list)
            # refer to https://github.com/rusty1s/pytorch_geometric/blob/eacb7d3e24a28aa50d7ae6d20a42676bc8ca1536
            # /torch_geometric/datasets/tu_dataset.py#L140
