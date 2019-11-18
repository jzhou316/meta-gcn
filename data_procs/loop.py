import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes


def add_self_loops_ey(edge_index, edge_y=None, node_y=None, num_nodes=None):
    """
    Add self-loops.
    The edges can optionally have labels (not necessary binary), in which case the loop edge label will be set
    according to the node label.
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    if edge_y is not None:
        assert node_y is not None
        assert len(node_y) == loop_index.size(1)
        edge_y = torch.cat([edge_y, node_y], dim=0)

    return edge_index, edge_y
