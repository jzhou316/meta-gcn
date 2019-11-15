#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python botnet_train.py --enc_sizes 1 1 1 1 1 1 1 1 --residual_hop 1 --lr 0.001 --epochs 50 --save_name chord_us_model_lay8_sz1_rh1_ep50.pt --bsz 2 --in_mem --devid 0
# CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 2 2 2 2 2 2 2 2 --residual_hop 1 --norm_method rw --lr 0.001 --epochs 50 --save_name chord_us_model_lay8_sz2_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0
# CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 16 16 32 32 64 64 128 128 --residual_hop 1 --lr 0.001 --epochs 50 --save_name chord_us_model_lay8_sz16_32_64_128_rh1_ep50.pt --bsz 2 --in_mem --devid 0

train_botnet () {
    CUDA_VISIBLE_DEVICES=$1 python botnet_train.py --in_channels 1 --enc_sizes 32 32 32 32 32 32 --residual_hop 1 --deg_norm sm --bias 0 --lr 0.001 --epochs 50 --save_name $2_no100k_ev10k_us_model_lay6_rh1_ep50.pt --bsz 2 --in_mem --devid 0 --data_train $2_no100k_ev10k_us_train.hdf5 --data_val $2_no100k_ev10k_us_val.hdf5 --data_test $2_no100k_ev10k_us_test.hdf5 --seed 0
echo
CUDA_VISIBLE_DEVICES=$1 python botnet_train.py --in_channels 1 --enc_sizes 32 32 32 32 32 32 32 32 --residual_hop 1 --deg_norm sm --bias 0 --lr 0.001 --epochs 50 --save_name $2_no100k_ev10k_us_model_lay8_rh1_ep50.pt --bsz 2 --in_mem --devid 0 --data_train $2_no100k_ev10k_us_train.hdf5 --data_val $2_no100k_ev10k_us_val.hdf5 --data_test $2_no100k_ev10k_us_test.hdf5 --seed 0
echo
CUDA_VISIBLE_DEVICES=$1 python botnet_train.py --in_channels 1 --enc_sizes 32 32 32 32 32 32 32 32 32 32 --residual_hop 1 --deg_norm sm --bias 0 --lr 0.001 --epochs 50 --save_name $2_no100k_ev10k_us_model_lay10_rh1_ep50.pt --bsz 2 --in_mem --devid 0 --data_train $2_no100k_ev10k_us_train.hdf5 --data_val $2_no100k_ev10k_us_val.hdf5 --data_test $2_no100k_ev10k_us_test.hdf5 --seed 0
echo
CUDA_VISIBLE_DEVICES=$1 python botnet_train.py --in_channels 1 --enc_sizes 32 32 32 32 32 32 32 32 32 32 32 32 --residual_hop 1 --deg_norm sm --bias 0 --lr 0.001 --epochs 50 --save_name $2_no100k_ev10k_us_model_lay12_rh1_ep50.pt --bsz 2 --in_mem --devid 0 --data_train $2_no100k_ev10k_us_train.hdf5 --data_val $2_no100k_ev10k_us_val.hdf5 --data_test $2_no100k_ev10k_us_test.hdf5 --seed 0
echo

CUDA_VISIBLE_DEVICES=$1 python botnet_train.py --in_channels 1 --enc_sizes 32 32 32 32 32 32 --residual_hop 1 --deg_norm rw --bias 0 --lr 0.001 --epochs 50 --save_name $2_no100k_ev10k_us_model_lay6_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0 --data_train $2_no100k_ev10k_us_train.hdf5 --data_val $2_no100k_ev10k_us_val.hdf5 --data_test $2_no100k_ev10k_us_test.hdf5 --seed 0
echo
CUDA_VISIBLE_DEVICES=$1 python botnet_train.py --in_channels 1 --enc_sizes 32 32 32 32 32 32 32 32 --residual_hop 1 --deg_norm rw --bias 0 --lr 0.001 --epochs 50 --save_name $2_no100k_ev10k_us_model_lay8_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0 --data_train $2_no100k_ev10k_us_train.hdf5 --data_val $2_no100k_ev10k_us_val.hdf5 --data_test $2_no100k_ev10k_us_test.hdf5 --seed 0
echo
CUDA_VISIBLE_DEVICES=$1 python botnet_train.py --in_channels 1 --enc_sizes 32 32 32 32 32 32 32 32 32 32 --residual_hop 1 --deg_norm rw --bias 0 --lr 0.001 --epochs 50 --save_name $2_no100k_ev10k_us_model_lay10_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0 --data_train $2_no100k_ev10k_us_train.hdf5 --data_val $2_no100k_ev10k_us_val.hdf5 --data_test $2_no100k_ev10k_us_test.hdf5 --seed 0
echo
CUDA_VISIBLE_DEVICES=$1 python botnet_train.py --in_channels 1 --enc_sizes 32 32 32 32 32 32 32 32 32 32 32 32 --residual_hop 1 --deg_norm rw --bias 0 --lr 0.001 --epochs 50 --save_name $2_no100k_ev10k_us_model_lay12_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0 --data_train $2_no100k_ev10k_us_train.hdf5 --data_val $2_no100k_ev10k_us_val.hdf5 --data_test $2_no100k_ev10k_us_test.hdf5 --seed 0
echo
}

train_botnet $1 $2
# train_botnet "0" "chord"
# train_botnet "0" "debru"
# train_botnet "0" "kadem"
# train_botnet "0" "leet"
