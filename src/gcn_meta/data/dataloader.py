from torch.utils.data import DataLoader

from torch_geometric.data import Batch


def collate_graph(graph_obj_list):
    """
    Collating function to form graph mini-batch.
    It takes in a list of GraphData objects and return a Batch.
    """
    batch = Batch.from_data_list(graph_obj_list)
    return batch


class GraphDataLoader(DataLoader):
    """
    Graph data loader, for a series of static graphs.

    Args:
        dataset (GraphDataset): graph dataset object
        batch_size (int, optional): batch size
        num_workers (int, optional): number of workers for multiple subprocesses
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_graph,
            num_workers=num_workers)


def collate_graph_dataset(graph_ds_list):
    """
    Collating function to form graph mini-batch.
    It takes in a list of Graph Dataset objects and return a Batch.
    """
    breakpoint()
    batch = Batch.from_data_list([gd[0] for gd in graph_ds_list])
    # each gd has only one graph data, and gd.data doesn't work (figure out why later)!
    return batch


class GraphDataLoaderDataset(DataLoader):
    """
    Graph data loader, for a series of static graph datasets.

    Args:
        dataset (GraphDataset): graph dataset object
        batch_size (int, optional): batch size
        num_workers (int, optional): number of workers for multiple subprocesses
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_graph_dataset,
            num_workers=num_workers)
