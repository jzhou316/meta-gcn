#!/bin/bash


CUDA_VISIBLE_DEVICES=2 python botnet_train.py --enc_sizes 32 32 32 32 32 32 --residual_hop 1 --norm_method None --edge_gate proj --lr 0.005 --epochs 50 --save_name leet_no100k_ev10k_us_model_lay6_rh1_eg_ep50.pt --bsz 2 --in_mem --devid 0 --data_train leet_no100k_ev10k_udr_slp_train.hdf5 --data_val leet_no100k_ev10k_udr_slp_val.hdf5 --data_test leet_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=2 python botnet_train.py --enc_sizes 32 32 32 32 32 32 32 32 --residual_hop 1 --norm_method None --edge_gate proj --lr 0.005 --epochs 50 --save_name leet_no100k_ev10k_us_model_lay8_rh1_eg_ep50.pt --bsz 2 --in_mem --devid 0 --data_train leet_no100k_ev10k_udr_slp_train.hdf5 --data_val leet_no100k_ev10k_udr_slp_val.hdf5 --data_test leet_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=2 python botnet_train.py --enc_sizes 32 32 32 32 32 32 32 32 32 32 --residual_hop 1 --norm_method None --edge_gate proj --lr 0.005 --epochs 50 --save_name leet_no100k_ev10k_us_model_lay10_rh1_eg_ep50.pt --bsz 2 --in_mem --devid 0 --data_train leet_no100k_ev10k_udr_slp_train.hdf5 --data_val leet_no100k_ev10k_udr_slp_val.hdf5 --data_test leet_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=2 python botnet_train.py --enc_sizes 32 32 32 32 32 32 32 32 32 32 32 32 --residual_hop 1 --norm_method None --edge_gate proj --lr 0.005 --epochs 50 --save_name leet_no100k_ev10k_us_model_lay12_rh1_eg_ep50.pt --bsz 2 --in_mem --devid 0 --data_train leet_no100k_ev10k_udr_slp_train.hdf5 --data_val leet_no100k_ev10k_udr_slp_val.hdf5 --data_test leet_no100k_ev10k_udr_slp_test.hdf5
echo
echo
