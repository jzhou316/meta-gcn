import torch
import torch.nn as nn

from .gcn_base_models import NodeModelAdditive
from .graph_attention import NodeModelAttention
from .graph_hard_attention import NodeModelHardAttention


class GCNMultiKernel(nn.Module):
    """
    Graph convolutional neural network with multiple kernels.
    This class only contains one such layer, without non-linear activations.

    Some features:
        - degree normalization for node features (based on out-degree)
        - edge gating mechanism when doing message passing
        - multiple kernels for the node neighborhood

    Args:
        num_kernel (int, optional): number of different GCN kernels.
        nodemodel (str, optional): node model name
        *args, **kwargs: arguments for the specific GCN node models. Could include the following:

        in_channels (int): input channels
        out_channels (int): output channels
        in_edgedim (int, optional): input edge feature dimension
        deg_norm (str, optional): method of (out-)degree normalization. Choose from [None, 'sm', 'rw']. Default: 'sm'.
            'sm': symmetric, better for undirected graphs. 'rw': random walk, better for directed graphs.
            Note that 'sm' for directed graphs might have some problems, when a target node does not have any out-degree.
        edge_gate (str, optional): method of applying edge gating mechanism. Choose from [None, 'proj', 'free'].
            Note that when set to 'free', should also provide `num_edges` as an argument (but then it can only work with
            fixed edge graph).
        aggr (str, optional): method of aggregating the neighborhood features. Choose from ['add', 'mean', 'max'].
            Default: 'add'.
        bias (bool, optional): whether to include bias vector in the model.
        **kwargs: could include `num_edges`, etc.

    Input:
        - x (torch.Tensor): input features, of size (N, C_in)
        - edge_index_K (list[torch.Tensor] or torch.Tensor): K or 1 different set of graph edges,
            each of size (2, E_k)
        - edge_attr_K (list[torch.Tensor] or torch.Tensor, optional): K or 1 different set of edge attributes/features,
            each of size (E_k, D_in). Default: None.
        - deg_K (list[torch.Tensor] or torch.Tensor, optional): K or 1 different set of node degrees,
            each of size (N,). Default: None.
        - edge_weight_K (list[torch.Tensor] or torch.Tensor, optional): K or 1 different set of edge weights,
            each of size (E_k,). Default: None.

    Output:
        - xo (torch.Tensor): updated node features, of size (N, C_out).

    where
        N: number of nodes
        E_k: number of edges for the k-th set of edges
        K: total number of kernels
        C_in/C_out: input/output number of channels
        D_in: input edge feature dimension
    """
    nodemodel_dict = {'additive': NodeModelAdditive,
                      'attention': NodeModelAttention,
                      'hardattention': NodeModelHardAttention}

    def __init__(self, *args, num_kernel=1, nodemodel='additive', kernel_combine='add', **kwargs):
        assert nodemodel in ['additive', 'attention', 'hardattention']
        assert kernel_combine in ['add', 'cat', 'mean']
        super(GCNMultiKernel, self).__init__()

        self.kernel_combine = kernel_combine

        self.node_models = nn.ModuleList([self.nodemodel_dict[nodemodel](*args, **kwargs) for k in range(num_kernel)])

    def reset_parameters(self):
        for net in self.node_models:
            net.reset_parameters()

    def forward(self, x, edge_index_K, edge_attr_K=None, deg_K=None, edge_weight_K=None, **kwargs):
        if isinstance(edge_index_K, torch.Tensor):
            edge_index_K = [edge_index_K]
        if isinstance(edge_attr_K, torch.Tensor):
            edge_attr_K = [edge_attr_K]
        if isinstance(deg_K, torch.Tensor):
            deg_K = [deg_K]
        if isinstance(edge_weight_K, torch.Tensor):
            edge_weight_K = [edge_weight_K]

        if edge_attr_K is None:
            edge_attr_K = [None] * len(edge_index_K)
        if deg_K is None:
            deg_K = [None] * len(edge_index_K)
        if edge_weight_K is None:
            edge_weight_K = [None] * len(edge_index_K)

        xo = 0
        for nm, edge_index, edge_attr, deg, edge_weight in \
                zip(self.node_models, edge_index_K, edge_attr_K, deg_K, edge_weight_K):
            if edge_index is not None:
                if self.kernel_combine == 'add':
                    xo += nm(x, edge_index, edge_attr, deg, edge_weight, **kwargs)
                elif self.kernel_combine == 'cat':
                    xo = torch.cat([xo, nm(x, edge_index, edge_attr, deg, edge_weight, **kwargs)], dim=1) if xo != 0 \
                        else nm(x, edge_index, edge_attr, deg, edge_weight, **kwargs)
                else:  # mean
                    if xo == 0:
                        count = 0
                    xo += nm(x, edge_index, edge_attr, deg, edge_weight, **kwargs)
                    count += 1

        if self.kernel_combine == 'mean':
            try:
                xo = xo / count
            except:
                print('This shouldn\'t happen...')

        return xo
