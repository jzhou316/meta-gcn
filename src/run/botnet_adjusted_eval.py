import argparse
import os
# import time

import torch
import torch.nn as nn
from tqdm import tqdm

# from misc import timeSince, logging
from metrics import *
from focal_loss import FocalLoss
import sys
sys.path.insert(0, '../data')
from dataset import DSTGraphDataset
from dataloader import DSTGraphLoader
sys.path.insert(0, '../models')
from dstgcn_model import DstGCNModel


##### some default parameters
data_dir = '/n/holylfs/LABS/tata_yu_rush_level1_data/TataProject/data_botnet'
data_name = 'chord_no100k_ev10k_us_test.hdf5'

## quick testing
# data_name = 'chord_no100k_ev10k_us_small.hdf5'

# in_memory = False
# batch_size = 1

devid = 0
# seed = 0

# norm_method = 'sm'
# if_focal_loss = False

threshold = 0.5

model_dir = './saved_models_nobias'
model_name = 'chord_no100k_ev10k_us_model_lay8_rh1_ep50.pt'
###########


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of a GCN model for node classification with adjustable threshold.')
    # general setting
    parser.add_argument('--devid', type=int, default=devid, help='device id; -1 for CPU')
#     parser.add_argument('--seed', type=int, default=seed, help='random seed')
    # data loading
    parser.add_argument('--data_dir', type=str, default=data_dir, help='directory to find the dataset')
    parser.add_argument('--data', type=str, default=data_name, help='file name of the evaluation dataset')
#     parser.add_argument('--bsz', type=int, default=batch_size, help='batch size')
    parser.add_argument('--in_mem', action='store_true', help='whether to load all the data into memory')
    # model
    parser.add_argument('--model_dir', type=str, default=model_dir, help='model directory')
    parser.add_argument('--model', type=str, default=model_name, help='saved model file name')
#     parser.add_argument('--norm_method', type=str, choices=['sm', 'rw'], default=norm_method, help='degree normalization method')
    parser.add_argument('--focal', action='store_true', help='whether to use focal loss or normal CrossEntropyLoss')
    # binary classification threshold
    parser.add_argument('--threshold', type=float, default=threshold, help='binary classification threshold on the predictive probability')
    args = parser.parse_args()
    return args


def eval_adjusted(model, data_loader, threshold=0.5, devid=-1, **kwargs):
    assert threshold > 0 and threshold < 1
    model.eval()
    loss_avg = 0
    acc = fpr = fnr = rec = prc = 0
    with torch.no_grad():
        for n, batch in tqdm(enumerate(data_loader)):        # here batch size is 1 for val and test data
            if devid > -1:
                batch.to(f'cuda:{devid}')
            x = model(batch.x[:, 0].view(-1, 1), batch.edge_index, deg_list=batch.x[:, 1], **kwargs)
            y = batch.y.long()
            loss = criterion(x, y)
            loss_avg += float(loss)
#             pred = x.argmax(dim=1)        # this is the case when threshold = 0.5
            pred = x[:, 1] > (x[:, 0] - torch.log(1 / torch.tensor(threshold) - 1))
            y = y.to(pred.dtype)
            acc += accuracy(pred, y)
            fpr += false_positive_rate(pred, y)
            fnr += false_negative_rate(pred, y)
            rec += recall(pred, y)
            prc += precision(pred, y)

    loss_avg = loss_avg / (n + 1)
    acc = acc / (n + 1)
    fpr = fpr / (n + 1)
    fnr = fnr / (n + 1)
    rec = rec / (n + 1)
    prc = prc / (n + 1)

    return loss_avg, acc, fpr, fnr, rec, prc


if __name__ == '__main__':
    args = parse_args()

    model = torch.load(os.path.join(args.model_dir, args.model), map_location='cpu')
    if args.devid > -1:
        model.to(f'cuda:{args.devid}')

    print('loading dataset...')
    dataset = DSTGraphDataset(os.path.join(args.data_dir, args.data), time_window=1, in_memory=args.in_mem)
    data_loader = DSTGraphLoader(dataset, batch_size=1)        # batch_size = 1 for correct average of evaluation metrics

    if not args.focal:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = FocalLoss(alpha=1, gamma=2)
    
    loss_avg, acc, fpr, fnr, rec, prc = eval_adjusted(model, data_loader, threshold=args.threshold, devid=args.devid)
    print(f'Evaluation --- loss: {loss_avg:.5f}, acc: {acc:.5f}, false positive rate: {fpr:.6f}, false negative rate: {fnr:.5f}')
    print(f'               recall (true positive rate): {rec:.5f}, precision: {prc:.5f}')

