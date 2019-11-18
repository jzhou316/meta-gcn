import torch
from torch_sparse.utils.unique import unique
from torch_geometric.utils.num_nodes import maybe_num_nodes


def sort_unique_edges(edge_index, num_nodes):
    """
    Modified from torch_sparse.coalesce, as a simplified version specifically for 'value' is None,
    and externally return sort indexes.
    """
    row, col = edge_index

    _, perm = unique(row * num_nodes + col)
    edge_index = torch.stack([row[perm], col[perm]], dim=0)

    return edge_index, perm


def to_undirected_ey(edge_index, edge_y=None, num_nodes=None):
    """
    Converts the graph given by :attr:`edge_index` to an undirected graph.
    The edges can optionally have labels (not necessary binary).
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, perm = sort_unique_edges(edge_index, num_nodes)

    if edge_y is not None:
        edge_y = torch.cat([edge_y, edge_y], dim=0)
        edge_y = edge_y[perm]

    return edge_index, edge_y
