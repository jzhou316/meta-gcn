import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops

from torch_geometric.nn.inits import glorot, zeros


class GCN(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, edge_din=None, improved=False, bias=True):
        super(GCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_din = edge_din
        self.improved = improved

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if edge_din is not None:
            self.weight_edge = Parameter(torch.Tensor(edge_din, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.edge_din is not None:
            glorot(self.weight_edge)
        zeros(self.bias)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index = add_self_loops(edge_index, num_nodes)
        loop_weight = torch.full(
            (num_nodes, ),
            1 if not improved else 2,
            dtype=edge_weight.dtype,
            device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        """"""
        x = torch.matmul(x, self.weight)
        edge_index, norm = GCN.norm(edge_index,
                                    x.size(0), edge_weight, self.improved,
                                    x.dtype)
        return self.propagate('add', edge_index, x=x, norm=norm, edge_attr=edge_attr)

    def message(self, x_j, norm, edge_attr=None):
        '''
        Transform source node feature and edge feature to a single feature on each source node before aggregation.
        x_j: (E_with_self_loop, C_out)
        edge_attr: (E, D_in)
        '''
        if edge_attr is None:
            x_jo = norm.view(-1, 1) * x_j
        else:
            assert self.edge_din is not None
            x_je = torch.matmul(edge_attr, self.weight_edge)    # size (E, C_out)
            x_je = torch.cat([x_je, x_je.new_zeros((x_j.size(0) - x_je.size(0), x_je.size(1)))], dim=0)    
                                                                # size (E_with_self_loop, C_out)
            x_jo = norm.view(-1, 1) * (x_j + x_je)
        return x_jo
            
    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels, self.edge_din)
    