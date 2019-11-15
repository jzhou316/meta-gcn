### A Meta-GCN Library
A lightweight unified framework with various graph convolutional neural networks (GCN) included.

There are tons of small model choices in GCN one can make, such as how to normalize
the feature to be aggregated from the neighborhood, what aggregation function to use,
whether to have different kernels for heterogeneous graphs, how to incorporate edge features,
whether to add residual connections, etc. Some of these variations have a name in the
literature, while some don't. But overall, they are all of the same model conceptually.   

In this library, different model choices are unified into the same high-level GCN framework,
with the idea lying in the center being *message passing*, which allows flexible exploration
and experiments on various graph data.

Implementation is based on PyTorch and PyTorch Geometric (PyG).

#### What's Included

Mostly what can be included in the message passing formulation of GCN.
The design is targeted at **directed graphs**, with undirected graphs as a special
case where each edge has two directions. It is also capable of dealing with
**multi-edges** and **heterogeneous** graphs, but all these distinctions are provided
in the data, whereas the model is agnostic.

For example, most of the GCN models require self-loops.
We do this in the data pre-processing step before feeding them into the model, which is different
from PyTorch Geometric where adding self-loops is part of the forward process, resulting in inefficiency for
large graphs. 

In particular, the following model variations are included:
- Message normalization constant with node degrees: *symmetric* (square root of the product of out degrees of
  both the sending node and the receiving node)
  or *random walk* (the out degree of the sending node)
- Neighborhood aggregation function: *sum*, *average*, *max*
- Different convolutional kernels: the original graph should be input in groups
- Edge gate: message along each edge is gated, one can choose whether the gate
  is calculated from the node features, or is a free parameter for each edge (in which case the
  graph structure should not change)
- Residual connections: for deeper networks, residual connection is important in good performance.
  One can choose the *residual hop* for how many layers apart to add the residual connections calculated
  from a linear transformation to match the correct feature dimension
- Graph attention with multi-head: attention over neighborhood. One can choose the *number of heads*,
  how to combine these heads (*sum* or *concatenate*), and what is the attention direction (*in* or *out*)
- Graph hard/discrete attention with multi-head: implemented with the Gumbel-Softmax trick
- Edge features: for each edge, add along with the node features to send
- Final output: node classification or graph classification (TODO)

These choices can be arbitrarily combined, and should provide a minimal working environment for
applying GCN to any customized graph data, with the task being node classification/regression/feature extraction,
or graph classification/embedding.

#### Data

Data should come in with HDF5 files (with consideration of large graphs).
We have customized data structure in HDF5 and data loaders. For anything
that's in other format, we have scripts transforming the data into our format.

Furthermore, the pre-processing steps usually include:
- Turn undirected graphs into directed format by adding edges with reverse directions
- Add self-loops in the graph
- split the data into train/validation/test sets (for customized graph data)

The data loader can load graphs in batches similarly to what is used in PyG.

#### Training and Evaluation

Generally with Adam, learning rate scheduler, and early stopping. Training logs and best models are saved
in specified directories. See our examples for more details.
