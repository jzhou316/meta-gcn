#!/bin/bash

# bot=chord    # chord, debru, kadem, leet
bot=$1

CUDA_VISIBLE_DEVICES=3 \
python train_botnet.py \
--devid 0 \
--seed 0 \
--in_mem 1 \
\
--data_dir ../../data/botnet/ev10k \
--data_train ${bot}_no100k_ev10k_us_train.hdf5 \
--data_val ${bot}_no100k_ev10k_us_val.hdf5 \
--data_test ${bot}_no100k_ev10k_us_test.hdf5 \
--save_dir ../saved_models/botnet_ev10k \
--save_name ${bot}_lay6_rh1_bs0_hatt_out_nh1_ep50.pt \
--bsz 2 \
\
--in_channels 1 \
--enc_sizes 32 32 32 32 32 32 \
--nheads 1 \
--act lrelu \
--residual_hop 1 \
--nodemodel hardattention \
--bias 0 \
--dropout 0.0 \
--final proj \
\
--att_dir out \
--att_act lrelu \
--att_dropout 0.0 \
--temperature 0.1 \
--sample 0 \
\
--lr 0.001 \
--weight_decay 5e-4 \
--epochs 50 \
