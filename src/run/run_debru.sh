#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 64 64 64 64 64 64 --residual_hop 1 --norm_method sm --lr 0.005 --epochs 50 --save_name debru_no100k_ev10k_us_model_lay6_sz64_rh1_ep50.pt --bsz 2 --in_mem --devid 0 --data_train debru_no100k_ev10k_udr_slp_train.hdf5 --data_val debru_no100k_ev10k_udr_slp_val.hdf5 --data_test debru_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 64 64 64 64 64 64 64 64 --residual_hop 1 --norm_method sm --lr 0.005 --epochs 50 --save_name debru_no100k_ev10k_us_model_lay8_sz64_rh1_ep50.pt --bsz 2 --in_mem --devid 0 --data_train debru_no100k_ev10k_udr_slp_train.hdf5 --data_val debru_no100k_ev10k_udr_slp_val.hdf5 --data_test debru_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 64 64 64 64 64 64 64 64 64 64 --residual_hop 1 --norm_method sm --lr 0.005 --epochs 50 --save_name debru_no100k_ev10k_us_model_lay10_sz64_rh1_ep50.pt --bsz 2 --in_mem --devid 0 --data_train debru_no100k_ev10k_udr_slp_train.hdf5 --data_val debru_no100k_ev10k_udr_slp_val.hdf5 --data_test debru_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 64 64 64 64 64 64 64 64 64 64 64 64 --residual_hop 1 --norm_method sm --lr 0.005 --epochs 50 --save_name debru_no100k_ev10k_us_model_lay12_sz64_rh1_ep50.pt --bsz 2 --in_mem --devid 0 --data_train debru_no100k_ev10k_udr_slp_train.hdf5 --data_val debru_no100k_ev10k_udr_slp_val.hdf5 --data_test debru_no100k_ev10k_udr_slp_test.hdf5
echo
echo
CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 64 64 64 64 64 64 --residual_hop 1 --norm_method rw --lr 0.005 --epochs 50 --save_name debru_no100k_ev10k_us_model_lay6_sz64_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0 --data_train debru_no100k_ev10k_udr_slp_train.hdf5 --data_val debru_no100k_ev10k_udr_slp_val.hdf5 --data_test debru_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 64 64 64 64 64 64 64 64 --residual_hop 1 --norm_method rw --lr 0.005 --epochs 50 --save_name debru_no100k_ev10k_us_model_lay8_sz64_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0 --data_train debru_no100k_ev10k_udr_slp_train.hdf5 --data_val debru_no100k_ev10k_udr_slp_val.hdf5 --data_test debru_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 64 64 64 64 64 64 64 64 64 64 --residual_hop 1 --norm_method rw --lr 0.005 --epochs 50 --save_name debru_no100k_ev10k_us_model_lay10_sz64_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0 --data_train debru_no100k_ev10k_udr_slp_train.hdf5 --data_val debru_no100k_ev10k_udr_slp_val.hdf5 --data_test debru_no100k_ev10k_udr_slp_test.hdf5
echo
CUDA_VISIBLE_DEVICES=1 python botnet_train.py --enc_sizes 64 64 64 64 64 64 64 64 64 64 64 64 --residual_hop 1 --norm_method rw --lr 0.005 --epochs 50 --save_name debru_no100k_ev10k_us_model_lay12_sz64_rh1_rw_ep50.pt --bsz 2 --in_mem --devid 0 --data_train debru_no100k_ev10k_udr_slp_train.hdf5 --data_val debru_no100k_ev10k_udr_slp_val.hdf5 --data_test debru_no100k_ev10k_udr_slp_test.hdf5
