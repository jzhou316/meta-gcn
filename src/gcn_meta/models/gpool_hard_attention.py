import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter_
from torch_scatter import scatter_max
from torch_geometric.utils import subgraph

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


def get_edge_select(x, edge_index, alpha, training=True, p_keep=None):
    """
    Based on the attention scores, select the important edges for graph pruning.
    Look at the attention scores before softmax, and only keep the argmax one.

    This would introduce a mismatch between training and testing, as in training, a node's feature is sent to
    all its neighbors, as opposed to being only sent to the one with max attention score.
    To compensate for this, for training, we could also look at the attention scores after softmax, and keep
    the ones that exceed a certain probability threshold; when the attention distribution is not close to one-hot,
    we keep all the neighbors (do not prune for this node's neighbors).

    Input:
        - x (torch.Tensor): node features, only used for getting `num_nodes` as x.size(0).
        - edge_index (torch.LongTensor): graph connectivity
        - alpha (torch.Tensor): attention scores, size (E, nheads).
            For training, this could be before softmax or after, and it must be after softmax with `p_keep` set.
            For testing, this should be the one-hot vector for each node.
        - training (bool): whether at training or testing
        - p_keep (None or float): if not None, at training time, instead of taking the argmax edge, take an edge
            based on its attention probability. Should be between 0 and 0.5.

    Output:
        - edge_select (torch.BoolTensor): edge selection mask, size (E,)
    """
    if training:
        # here alpha is the attention scores
        if p_keep is None:
            # could use attention scores before or after softmax
            # attention direction is fixed at 'out'
            out, argmax = scatter_max(alpha, edge_index[0], dim=0, dim_size=x.size(0), fill_value=-1e16)
            # both 'out' and 'argmax' are of size (N, nheads)
            edge_select = torch.zeros(alpha.size(0), dtype=torch.bool)
            edge_select[argmax.view(-1)] = 1    # looking at all heads
        else:
            # should only use attention scores after softmax!
            # if anything >= p_keep, select that edge; otherwise keep everything (no selection)
            # === first check the max sending probability for each node
            out, argmax = scatter_max(alpha, edge_index[0], dim=0, dim_size=x.size(0), fill_value=-1e16)
            # both 'out' and 'argmax' are of size (N, nheads)
            # === for any node, if the max is smaller than p_keep, keep all the edges there
            # look at all heads: if one head has all edges, then we keep all
            node_keep = (out < p_keep).sum(dim=1) > 0    # size (N,)
            edge_keep = node_keep[edge_index[0]]    # size (E,)
            # === select edges if one attention score is extreme or all are non-extreme
            edge_select = (alpha >= p_keep).sum(dim=1) > 0 | edge_keep
    else:
        # here alpha is the one-hot vector for each node
        edge_select = alpha.sum(dim=1) > 0  # looking at all heads
    return edge_select


def prune(x, edge_index, edge_select, relabel=False, batch_slices_x=None):
    """
    Prune the graph based on selected edges.

    Input:
        - x (torch.Tensor): node features, size (N, C)
        - edge_index (torch.LongTensor): graph connectivity, size (2, E)
        - edge_select (torch.BoolTensor): edge selection mask, size (E,)
        - relabel (bool): whether to relabel the nodes. If False, the node features of the pruned nodes will be
            set to zeros. If True, the node feature matrix will also be pruned.
        - batch_slices_x (list[int]): index slices for node features of different graphs in the batch.

    Output:
        - x (torch.Tensor): pruned node features
        - edge_index (torch.LongTensor): pruned edge indexes
        - nodes (torch.LongTensor): selected node list, sorted
        - batch_slices_x_pruned (list[int]): batch_slices_x for pruned node feature matrix.
    """
    nodes = edge_index[1, edge_select].unique()    # selected nodes
    if relabel:
        assert batch_slices_x is not None
        # relabelling process follows the original node id orders
        edge_index, _ = subgraph(nodes, edge_index, relabel_nodes=True, num_nodes=x.size(0))
        x = x[nodes, :]    # prune the node features
        # need to recalculate the batch_slices_x
        batch_slices_x_pruned = [0]
        for end in batch_slices_x[1:]:
            reduced_num = (nodes < end).sum().item()
            batch_slices_x_pruned.append(end - reduced_num)
    else:
        edge_index, _ = subgraph(nodes, edge_index, relabel_nodes=False, num_nodes=x.size(0))
        x[nodes, :] = 0    # mask out the unselected node features by 0
        batch_slices_x_pruned = batch_slices_x
    return x, edge_index, nodes, batch_slices_x_pruned


class VotePoolLayer(nn.Module):
    """
    Graph pooling layer by "voting" or "walking", i.e. using hard attention with out direction so that every node selects one node to
    send out its feature, towards aggregated higher level representation.
    """
    def __init__(self, in_channels, out_channels, nheads=1, att_act='none', att_dropout=0, att_combine='cat',
                 att_dir='out', bias=False, aggr='add', temperature=0.1, sample=False, p_keep=None, relabel=False, **kwargs):
        assert att_act in ['none', 'lrelu', 'relu']
        assert att_combine in ['cat', 'add', 'mean']
        assert att_dir == 'out'    # dummy
        super().__init__()

        self.aggr = aggr

        # for graph pruning
        self.p_keep = p_keep
        self.relabel = relabel

        self.nheads = nheads
        if att_combine == 'cat':
            self.out_channels_1head = out_channels // nheads
            assert self.out_channels_1head * nheads == out_channels, 'out_channels should be divisible by nheads'
        else:
            self.out_channels_1head = out_channels

        self.att_combine = att_combine
        self.att_dir = att_dir

        self.temperature = temperature
        self.sample = sample

        if att_combine == 'cat':
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        else:  # 'add' or 'mean':
            self.weight = Parameter(torch.Tensor(in_channels, out_channels * nheads))
        self.att_weight = Parameter(torch.Tensor(1, nheads, 2 * self.out_channels_1head))
        self.att_act = activation(att_act)
        self.att_dropout = nn.Dropout(p=att_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, batch_slices_x, attn_store=None):
        x = torch.mm(x, self.weight).view(-1, self.nheads, self.out_channels_1head)  # size (N, n_heads, C_out_1head)

        # lift the features to source and target nodes, size (E, nheads, C_out_1head) for each
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])

        # calculate attention coefficients, size (E, nheads)
        alpha = self.att_act((torch.cat([x_j, x_i], dim=-1) * self.att_weight).sum(dim=-1))

        if self.training:
            # gumbel-softmax trick, size (E, nheads)
            gumbel_noise = gumbel_samples(alpha)
            alpha = (alpha + gumbel_noise) / self.temperature

            # softmax over each node's neighborhood, size (E, nheads)
            if self.att_dir == 'out':
                # random walk
                alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))
            else:
                # attend over nodes that all points to the current one
                alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))

            # dropout on attention coefficients
            # (which means that during training, the neighbors are stochastically sampled)
            alpha = self.att_dropout(alpha)
        else:
            # categorical distribution: sample directly or take the max
            if self.sample:
                # still use the gumbel-softmax trick to find the argmax for sampling
                gumbel_noise = gumbel_samples(alpha)
                alpha = alpha + gumbel_noise    # size (E, nheads)
            # otherwise, directly take max according to alpha
            if self.att_dir == 'out':
                out, argmax = scatter_max(alpha, edge_index[0], dim=0, dim_size=x.size(0), fill_value=-1e16)
                # fill_value=-1e16 is necessary for scatter_max, otherwise it doesn't work correctly
                # similarly, setting fill_value=1e16 is necessary for scatter_min
            else:
                out, argmax = scatter_max(alpha, edge_index[1], dim=0, dim_size=x.size(0), fill_value=-1e16)
            # both 'out' and 'argmax' are of size (N, nheads)
            alpha_1hot = torch.zeros_like(alpha)
            alpha_1hot = alpha_1hot.t().reshape(-1)
            # set the right place to sample
            alpha_1hot[(argmax + alpha.size(0) * torch.arange(alpha.size(1)).to(argmax)).view(-1)] = 1.0
            alpha_1hot = alpha_1hot.reshape(alpha.size(1), -1).t()
            alpha = alpha_1hot    # size (E, nheads)
        ''' 
        # check attention entropy
        if self.att_dir == 'out':
            entropy = scatter_('add', -alpha * torch.log(alpha + 1e-16), edge_index[0], dim_size=x.size(0))
        else:    # size (N, nheads)
            entropy = scatter_('add', -alpha * torch.log(alpha + 1e-16), edge_index[1], dim_size=x.size(0))
        breakpoint()
        entropy = entropy[deg > 100, :].mean()
        entropy_max = (torch.log(deg[deg > 100] + 1e-16)).mean()
        print(f'average attention entropy {entropy.item()} (average max entropy {entropy_max.item()})')
        '''
        # normalize messages on each edges with attention coefficients
        x_j = x_j * alpha.view(-1, self.nheads, 1)

        # aggregate features to nodes, resulting in size (N, n_heads, C_out_1head)
        x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))

        # combine multi-heads, resulting in size (N, C_out)
        if self.att_combine == 'cat':
            x = x.view(-1, self.out_channels)
        elif self.att_combine == 'add':
            x = x.sum(dim=1)
        else:
            x = x.mean(dim=1)

        # add bias
        if self.bias is not None:
            x = x + self.bias

        if attn_store is not None:    # attn_store is a callback list in case we want to get the attention scores out
            attn_store.append(alpha)

        # prune the graph based on out hard attention
        # with torch.no_grad():
        edge_select = get_edge_select(x, edge_index, alpha, training=self.training, p_keep=self.p_keep)
        x, edge_index, nodes, batch_slices_x = prune(x, edge_index, edge_select,
                                                     relabel=self.relabel, batch_slices_x=batch_slices_x)

        return x, edge_index, nodes, batch_slices_x

    @property
    def num_parameters(self):
        if not hasattr(self, 'num_para'):
            self.num_para = sum([p.nelement() for p in self.parameters()])
        return self.num_para

    def __repr__(self):
        return ('{} (in_channels: {}, out_channels: {}, nheads: {}, att_activation: {},'
                'att_dropout: {}, att_combine: {}, att_dir: {}, temperature: {}, sample: {}, '
                'pruning_p: {}, pruning_relabel: {} | number of parameters: {})').format(
                self.__class__.__name__, self.in_channels, self.out_channels, self.nheads, self.att_act,
                self.att_dropout.p, self.att_combine, self.att_dir, self.temperature, self.sample,
                self.p_keep, self.relabel, self.num_parameters)


class SATOutLayer(nn.Module):
    """
    Global output self-attention layer for the whole graph, for tasks such as graph classification.
    Use self-attention to aggregate the node features to output a single vector for the whole graph.
    """
    def __init__(self, channels, num_classes):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.self_attention = nn.Linear(channels, 1, bias=True)
        self.proj = nn.Linear(channels, num_classes, bias=True)

    def reset_parameters(self):
        self.self_attention.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x, batch_slices_x):
        if len(batch_slices_x) == 2:
            # only one graph in the batch
            x = x[x.sum(dim=1) > 1e-20]    # filter out the zero rows, as have been pruned during the process
            attn_weights = nn.functional.softmax(self.self_attention(x), dim=0)
            out = self.proj((x * attn_weights.view(-1, 1)).sum(dim=0, keepdim=True))    # size (1, num_classes)
        else:
            # more than one graphs in the batch
            x_batch = [x[i:j] for (i, j) in zip(batch_slices_x, batch_slices_x[1:])]
            x_batch = pad_sequence(x_batch, batch_first=True, padding_value=0)  # size (batch_size, max_num_nodes, hid)
            # masks out padding nodes in the batch, as well as the pruned nodes
            mask = x_batch.sum(dim=-1) > 1e-20    # size (batch_size, max_num_nodes)
            # masked attention
            attn_scores = self.self_attention(x_batch)    # size (batch_size, max_num_nodes, 1)
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(2), float('-inf'))
            attn_weights = nn.functional.softmax(attn_scores, dim=1)
            out = (x_batch * attn_weights).sum(dim=1)    # size (batch_size, hid)
            out = self.proj(out)    # size (batch_size, num_classes)
        return out


class VotePoolModel(nn.Module):
    """
    Graph pooling model composed of several layers and an output layer for graph classification.
    """
    def __init__(self, in_channels, enc_sizes, num_classes, non_linear='relu', residual=True, dropout=0.0, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.enc_sizes = [in_channels, *enc_sizes]
        self.num_layers = len(self.enc_sizes) - 1
        self.num_classes = num_classes
        self.residual = residual

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

        # build up the layers
        self.gpool_net = nn.ModuleList([VotePoolLayer(in_c, out_c, nheads=nh, **kwargs)
                                        for in_c, out_c, nh in zip(self.enc_sizes, self.enc_sizes[1:], self.nheads)])
        self.dropout = nn.Dropout(dropout)
        if residual:
            self.residual_net = nn.ModuleList([nn.Linear(in_c, out_c, bias=True)
                                               for in_c, out_c in zip(self.enc_sizes, self.enc_sizes[1:])])
            self.num_residuals = len(self.residual_new)
        self.non_linear = activation(non_linear)
        self.out_net = SATOutLayer(enc_sizes[-1], num_classes)

    def reset_parameters(self):
        for net, rnet in zip(self.gpool_net, self.residual_net):
            net.reset_parameters()
            rnet.reset_parameters()
        self.out_net.reset_parameters()

    def forward(self, x, edge_index, batch_slices_x):
        remaining_nodes = []
        for n, net in enumerate(self.gpool_net):
            xo, edge_index, nodes, batch_slices_x = net(x, edge_index, batch_slices_x)
            xo = self.dropout(xo)
            if self.residual:
                # xo = self.non_linear(xo)
                # residual connections
                xr = self.residual_net[n](x)
                if net.relabel:
                    xr = xr[nodes]
                else:
                    xr[nodes] = 0
                x = self.non_linear(xo + xr)
            else:
                x = self.non_linear(xo)

            remaining_nodes.append(nodes)

        x = self.out_net(x, batch_slices_x)    # size (num_graphs, num_classes)
        return x, remaining_nodes
