import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import degree
import torch_geometric.transforms as T


class NormalizedDegree(object):
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


def get_dataset(name, sparse=True, x_deg=True, add_sl=False, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
#    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset = TUDataset(path, name)
    dataset.data.edge_attr = None

    if add_sl:
        '''this doesn't work, as the data are copied when indexed out
        # for data in dataset:
        #     data.edge_index, _ = add_remaining_self_loops(data.edge_index)
        '''
        data_list = []
        for data in dataset:
            data.edge_index, _ = add_remaining_self_loops(data.edge_index)
            data_list.append(data)
        dataset.data, dataset.slices = dataset.collate(data_list)
        # refer to https://github.com/rusty1s/pytorch_geometric/blob/eacb7d3e24a28aa50d7ae6d20a42676bc8ca1536
        # /torch_geometric/datasets/tu_dataset.py#L140

    if dataset.data.x is None:
        if x_deg:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)
        else:
            dataset.transform = NodeFeatureOnes()

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset
