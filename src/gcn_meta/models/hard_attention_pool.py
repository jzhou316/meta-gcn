import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter_
from torch_geometric.nn.pool.topk_pool import filter_adj

from .common import activation, softmax


def gumbel_samples(base):
    """
    Return iid samples from Gumbel distribution. 'size', 'dtype', and 'device' are from the `base` tensor.
    """
    noise = torch.rand(base.size()).to(base)    # .to(other) will convert dtype and device
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return noise


class HardPooling(nn.Module):
    def __init__(self, in_channels, att_act='none', att_dropout=0.0,
                 aggr='add', bias=False, temperature=0.1, sample=False, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.aggr = aggr

        self.temperature = temperature
        self.sample = sample

        self.att_weight = Parameter(torch.Tensor(1, 2 * in_channels))    # or use a linear layer with bias
        self.att_act = activation(att_act)
        self.att_dropout = nn.Dropout(p=att_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, batch, edge_attr=None, attn_store=None):
        # lift the features to source and target nodes, size (E, C_in) for each
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])

        # calculate attention coefficients, size (E, 1)
        alpha = self.att_act((torch.cat([x_j, x_i], dim=-1) * self.att_weight).sum(dim=-1, keepdim=True))

        if self.training:
            # gumbel-softmax trick, size (E, 1)
            gumbel_noise = gumbel_samples(alpha)
            alpha = (alpha + gumbel_noise) / self.temperature

            # softmax over each node's neighborhood, size (E, 1)
            # random walk
            alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))

            # dropout on attention coefficients
            # (which means that during training, the neighbors are stochastically sampled)
            alpha = self.att_dropout(alpha)
        else:
            # categorical distribution: sample directly or take the max
            if self.sample:
                # still use the gumbel-softmax trick to find the argmax for sampling
                gumbel_noise = gumbel_samples(alpha)
                alpha = alpha + gumbel_noise  # size (E, 1)
            # otherwise, directly take max according to alpha
            out, argmax = scatter_max(alpha, edge_index[0], dim=0, dim_size=x.size(0), fill_value=-1e16)
            # fill_value=-1e16 is necessary for scatter_max, otherwise it doesn't work correctly
            # similarly, setting fill_value=1e16 is necessary for scatter_min

            # both 'out' and 'argmax' are of size (N, 1)
            alpha_1hot = torch.zeros_like(alpha)
            alpha_1hot = alpha_1hot.t().reshape(-1)
            # set the right place to sample
            alpha_1hot[(argmax + alpha.size(0) * torch.arange(alpha.size(1)).to(argmax)).view(-1)] = 1.0
            alpha_1hot = alpha_1hot.reshape(alpha.size(1), -1).t()
            alpha = alpha_1hot  # size (E, 1)

        # normalize messages on each edges with attention coefficients
        x_j = x_j * alpha.view(-1, 1)

        # aggregate features to nodes, resulting in size (N, C_in)
        x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))

        # add bias
        if self.bias is not None:
            x = x + self.bias

        if attn_store is not None:  # attn_store is a callback list in case we want to get the attention scores out
            attn_store.append(alpha)

        # prune the graph based on out hard attention
        score = x.new_zeros(x.shape[0])
        if self.training:
            perm = torch.arange(x.size(0), device=x.device)
            pass
        else:
            edge_select = (alpha > 0).squeeze()
            perm = edge_index[1, edge_select].unique()  # selected nodes
            x = x[perm]
            batch = batch[perm]
            edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                               num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score
