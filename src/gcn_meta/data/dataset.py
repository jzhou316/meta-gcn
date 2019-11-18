import h5py
import deepdish as dd
from torch.utils.data import Dataset

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
