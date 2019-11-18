#!/bin/bash

# preprocess the graph dataset for P2P botnet detection, stored in a HDF5 file
# graphs are directed, with no self-loops, and no node degrees stored
# 1. add undirected and self-loop edges
# 2. calculate node (out-)degrees and store in node features x 
# 3. split data into train/val/test sets

set -e

preproc () {
    data_dir=/n/holylfs/LABS/tata_yu_rush_level1_data/xuzhiying/public_data_processed
    save_dir=../data/botnet/ev10k
    python data_add_edges_ey.py $1.hdf5 --data_dir $data_dir --save_dir $data_dir
    python data_add_degree.py $1_us.hdf5 --data_dir $data_dir --append2x 1
    python data_split.py $1_us.hdf5 --data_dir $data_dir --save_dir $save_dir
}

# preproc chord/chord_no100k_ev10k        # note: don't include file ext
preproc $1

