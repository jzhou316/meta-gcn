#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
python train_citation.py \
--devid 0 \
--seed 0 \
\
--dataset Cora \
--save_name temp_model.pt \
\
--enc_sizes 32 32 7 \
--nheads 2 2 1 \
--residual_hop 1 \
--nodemodel attention \
--bias 0 \
--dropout 0.6 \
--final none \
--lr 0.005 \
--epochs 500 \
--att_dir in \
--att_act lrelu \
--att_dropout 0.6 \
--temperature 0.5 \
--sample 0
