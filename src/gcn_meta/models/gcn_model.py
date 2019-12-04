import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .gcn_multi_kernel import GCNMultiKernel
from .common import activation


class GCNModel(nn.Module):
    """
    Graph convolutional model, composed of several GCN layers, residual connections, and final output layers.

    TODO:
        add doc
        different end tasks, e.g. pred_on='node' or 'graph', etc.
        Or separate out another module for the final output task.
    """

    def __init__(self, in_channels, enc_sizes, num_classes, non_linear='relu', non_linear_layer_wise='relu',
                 residual_hop=None, dropout=0.5, final_layer_config=None, final_type='none', pred_on='node', **kwargs):
        assert final_type in ['none', 'proj']
        assert pred_on in ['node', 'graph']
        super().__init__()

        self.in_channels = in_channels
        self.enc_sizes = [in_channels, *enc_sizes]
        self.num_layers = len(self.enc_sizes) - 1
        self.num_classes = num_classes
        self.residual_hop = residual_hop
        self.non_linear_layer_wise = non_linear_layer_wise
        self.final_type = final_type
        self.pred_on = pred_on

        # allow different layers to have different attention heads
        # particularly for the last attention layer to be directly the output layer
        if 'nheads' in kwargs:
            if isinstance(kwargs['nheads'], int):
                self.nheads = [kwargs['nheads']] * self.num_layers
            elif isinstance(kwargs['nheads'], list):
                self.nheads = kwargs['nheads']
                assert len(self.nheads) == self.num_layers
            else:
                raise ValueError
            del kwargs['nheads']
        else:
            # otherwise just a placeholder for 'nheads'
            self.nheads = [1] * self.num_layers

        if final_layer_config is None:
            self.gcn_net = nn.ModuleList([GCNLayer(in_c, out_c, nheads=nh, non_linear=non_linear_layer_wise, **kwargs)
                                          for in_c, out_c, nh in zip(self.enc_sizes, self.enc_sizes[1:], self.nheads)])
        else:
            assert isinstance(final_layer_config, dict)
            self.gcn_net = nn.ModuleList([GCNLayer(in_c, out_c, nheads=nh, non_linear=non_linear_layer_wise, **kwargs)
                                          for in_c, out_c, nh in zip(self.enc_sizes[:-2],
                                                                     self.enc_sizes[1:-1],
                                                                     self.nheads[:-1])])
            kwargs.update(final_layer_config)    # this will update with the new values in final_layer_config
            self.gcn_net.append(GCNLayer(self.enc_sizes[-2], self.enc_sizes[-1], nheads=self.nheads[-1],
                                         non_linear=non_linear_layer_wise, **kwargs))

        self.dropout = nn.Dropout(dropout)

        if residual_hop is not None and residual_hop > 0:
            self.residuals = nn.ModuleList([nn.Linear(self.enc_sizes[i], self.enc_sizes[j])
                                            for i, j in zip(range(0, len(self.enc_sizes), residual_hop),
                                                            range(residual_hop, len(self.enc_sizes), residual_hop))])
            self.non_linear = activation(non_linear)
            self.num_residuals = len(self.residuals)

        if self.final_type == 'none':
            self.final = nn.Identity()
        elif self.final_type == 'proj':
            self.final = nn.Linear(self.enc_sizes[-1], num_classes)
        else:
            raise ValueError

    def reset_parameters(self):
        for net in self.gcn_net:
            net.reset_parameters()
        if self.residual_hop is not None:
            for net in self.residuals:
                net.reset_parameters()
        if self.final_type != 'none':
            self.final.reset_parameters()

    def forward(self, x, edge_index_K, edge_attr_K=None, deg_K=None, edge_weight_K=None, **kwargs):
        xr = None
        add_xr_at = -1
        for n, net in enumerate(self.gcn_net):
            # pass to a GCN layer with non-linear activation
            xo = net(x, edge_index_K, edge_attr_K, deg_K, edge_weight_K, **kwargs)
            xo = self.dropout(xo)
            # deal with residual connections
            if self.residual_hop is not None and self.residual_hop > 0:
                if n % self.residual_hop == 0 and (n // self.residual_hop) < self.num_residuals:
                    xr = self.residuals[n // self.residual_hop](x)
                    add_xr_at = n + self.residual_hop - 1
                if n == add_xr_at:
                    if n < self.num_layers - 1:    # before the last layer
                        # non_linear is applied both after each layer and after residual sum
                        xo = self.non_linear(xo + xr)
                    else:    # the last layer (potentially the output layer)
                        # no non_linear is important for binary classification since this is to be passed to sigmoid
                        # function to calculate loss, and ReLu will directly kill all the negative parts
                        xo = xo + xr
            x = xo
        # size of x: (B * N, self.enc_sizes[-1]) -> (B * N, num_classes)
        x = self.final(x)

        # graph level pooling for graph classification
        # use mean pooling here
        if self.pred_on == 'graph':
            assert 'batch_slices_x' in kwargs
            batch_slices_x = kwargs['batch_slices_x']
            if len(batch_slices_x) == 2:
                # only one graph in the batch
                x = x.mean(dim=0, keepdim=True)    # size (1, num_classes)
            else:
                # more than one graphs in the batch
                x_batch, lengths = zip(*[(x[i:j], j - i) for (i, j) in zip(batch_slices_x, batch_slices_x[1:])])
                x_batch = pad_sequence(x_batch, batch_first=True,
                                       padding_value=0)  # size (batch_size, max_num_nodes, num_classes)
                x = x_batch.sum(dim=1) / x_batch.new_tensor(lengths)    # size (batch_size, num_classes)

        return x


class GCNLayer(nn.Module):
    """
    Graph convolutional layer. A wrapper of the multi-kernel graph convolutional operator.
    Takes in a static graph and update the node features (maybe also edge features).

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        in_edgedim (int, optional): input edge feature dimension
        deg_norm (str, optional): method of (out-)degree normalization. Choose from [None, 'sm', 'rw']. Default: 'sm'.
            'sm': symmetric, better for undirected graphs. 'rw': random walk, better for directed graphs.
            Note that 'sm' for directed graphs might have some problems, when a target node does not have any out-degree.
        edge_gate (str, optional): method of apply edge gating mechanism. Choose from [None, 'proj', 'free'].
            Note that when set to 'free', should also provide `num_edges` as an argument (but then it can only work with
            fixed edge graph).
        aggr (str, optional): method of aggregating the neighborhood features. Choose from ['add', 'mean', 'max'].
            Default: 'add'.
        bias (bool, optional): whether to include bias vector in the model.
        num_kernel (int, optional): number of different GCN kernels.
        nodemodel (str, optional): node model name
        non_linear (str, optional): non-linear activation function
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

    TODO:
        consider batch size
    """

    def __init__(self, in_channels, out_channels, in_edgedim=None, deg_norm='sm', edge_gate=None, aggr='add',
                 bias=True, num_kernel=1, nodemodel='additive', non_linear='relu', **kwargs):
        super().__init__()
        self.gcn = GCNMultiKernel(in_channels,
                                  out_channels,
                                  in_edgedim,
                                  deg_norm=deg_norm,
                                  edge_gate=edge_gate,
                                  aggr=aggr,
                                  bias=bias,
                                  num_kernel=num_kernel,
                                  nodemodel=nodemodel,
                                  **kwargs)
        self.non_linear = activation(non_linear)

    def reset_parameters(self):
        self.gcn.reset_parameters()

    def forward(self, x, edge_index_K, edge_attr_K=None, deg_K=None, edge_weight_K=None, **kwargs):
        xo = self.gcn(x, edge_index_K, edge_attr_K, deg_K, edge_weight_K, **kwargs)
        xo = self.non_linear(xo)
        return xo
