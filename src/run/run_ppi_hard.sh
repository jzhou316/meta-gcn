#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
python train_ppi.py \
--devid 0 \
--seed 0 \
\
--dataset PPI \
--bsz 1 \
--save_name temp_model.pt \
\
--enc_sizes 1024 1024 121 \
--nheads 8 8 1 \
--act lrelu \
--residual_hop 1 \
--nodemodel hardattention \
--bias 1 \
--dropout 0.0 \
--final none \
\
--att_dir in \
--att_act lrelu \
--att_dropout 0.0 \
--temperature 0.1 \
--sample 0 \
\
--lr 0.005 \
--weight_decay 0 \
--epochs 100 \
