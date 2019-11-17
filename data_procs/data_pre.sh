#!/bin/bash

# preprocess the graph dataset for P2P botnet detection, stored in a HDF5 file
# graphs are directed, with no self-loops, and no node degrees stored
# 1. add undirected and self-loop edges
# 2. calculate node (out-)degrees and store in node features x 
# 3. split data into train/val/test sets

preproc () {
    python data_add_edges.py $1.hdf5
    python data_add_degree.py $1_us.hdf5
    python data_split.py $1_us.hdf5
}

# preproc chord/chord_no100k_ev1k        # note: don't include file ext
preproc $1

