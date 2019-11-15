#!/bin/bash


adj_eval () {
    threshold=${3:-0.5}
    for m in rh1 rh1_rw
    do
#        for (( ly=6; ly<=12; ly=ly+2 ))
        for ly in {6..12..2}
        do
            echo "botnet: $2, number of layers: $ly, model variation: $m"
            python botnet_adjusted_eval.py --data $2_no100k_ev10k_us_test.hdf5 --model $2_no100k_ev10k_us_model_lay${ly}_${m}_ep50.pt --devid $1 --in_mem --threshold $threshold
            echo
        done
    done
}

# adj_eval 0 chord
# adj_eval 0 chord 0.5
adj_eval $1 $2 $3
