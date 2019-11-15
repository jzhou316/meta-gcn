#!/bin/bash

# check model log tails for information

check_mlog_tail () {
    for m in rh1 rh1_rw
    do
        for ly in {6..12..2}
        do
            tail saved_models_nobias/$1_no100k_ev10k_us_model_lay${ly}_${m}_ep50.log
            echo
        done
    done
}

#check_mlog_tail chord
check_mlog_tail $1
