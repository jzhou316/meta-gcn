import argparse
import os
import time

import torch
import torch.nn as nn

from misc import timeSince, logging, print_cuda_mem
from metrics import *
from earlystop import EarlyStopping
from focal_loss import FocalLoss
import sys
sys.path.insert(0, '../data')
from dataset import DstGraphDatasetLarge
from dataloader import DstGraphLoader
sys.path.insert(0, '../models')
# from dstgcn_model import DstGCNModel
from dstgcn_model_time import DstGCNModel


##### some default parameters
botnet = 'kademlia'        # 'debruijn', 'kademlia', 'leet'
data_dir = f'/n/holylfs/LABS/tata_yu_rush_level1_data/xuzhiying/public_data_processed/{botnet}_time/{botnet}_10k_evils'
# data_dir = '/n/holylfs/LABS/tata_yu_rush_level1_data/TataProject/data_botnet'
data_train_name = [f'{botnet}_time_{i}_evils_0.2.hdf5' for i in range(14)]
# data_train_path = os.path.join(data_dir, data_train_name)

data_val_name = f'{botnet}_time_{14}_evils_0.2.hdf5'
# data_val_path = os.path.join(data_dir, data_val_name)

data_test_name = f'{botnet}_time_{15}_evils_0.2.hdf5'
# data_test_path = os.path.join(data_dir, data_test_name)

## quick testing
# data_train_path = '/media/work/TataProject/data_chord/chord_graphs_small.hdf5'
# data_val_path = '/media/work/TataProject/data_chord/chord_graphs_small.hdf5'
# data_test_path = '/media/work/TataProject/data_chord/chord_graphs_small.hdf5'
# data_val_path = '/n/holylfs/LABS/tata_yu_rush_level1_data/TataProject/data_chord/chord_graphs_small.hdf5'
# data_test_path = '/n/holylfs/LABS/tata_yu_rush_level1_data/TataProject/data_chord/chord_graphs_small.hdf5'

# in_memory = False
batch_size = 1
time_window = 10

devid = 0
seed = 0

in_channels = 1
enc_sizes = [32, 32, 32, 32, 32, 32, 32, 32]
residual_hop = 0
edge_gate = 'None'
num_classes = 2
norm_method = 'sm'

learning_rate = 0.001
# grad_max_norm = 5

num_epochs = 5
save_dir = './saved_models_time'
save_name = 'temp_model.pt'


def parse_args():
    parser = argparse.ArgumentParser(description='Training a GCN model.')
    # general setting
    parser.add_argument('--devid', type=int, default=devid, help='device id; -1 for CPU')
    parser.add_argument('--seed', type=int, default=seed, help='random seed')
    parser.add_argument('--logmode', type=str, default='w', help='logging file mode')
    # data loading
    parser.add_argument('--data_dir', type=str, default=data_dir, help='directory to find the dataset')
    parser.add_argument('--data_train', type=str, nargs='*', default=data_train_name, help='file name of the training dataset')
    parser.add_argument('--data_val', type=str, default=data_val_name, help='file name of the validation dataset')
    parser.add_argument('--data_test', type=str, default=data_test_name, help='file name of the test dataset')
    parser.add_argument('--bsz', type=int, default=batch_size, help='batch size')
    parser.add_argument('--tw', type=int, default=time_window, help='time_window')
    parser.add_argument('--in_mem', action='store_true', help='whether to load all the data into memory')
    # model
    parser.add_argument('--in_channels', type=int, default=in_channels, help='input node feature size')
    parser.add_argument('--enc_sizes', type=int, nargs='*', default=enc_sizes, help='encoding node feature sizes')
    parser.add_argument('--residual_hop', type=int, default=residual_hop, help='residual per # layers')
    parser.add_argument('--edge_gate', type=str, choices=['None', 'proj', 'free'], default=edge_gate, help='types of edge gate')
    parser.add_argument('--n_classes', type=int, default=num_classes, help='number of classes for the output layer')
    parser.add_argument('--norm_method', type=str, choices=['sm', 'rw'], default=norm_method, help='degree normalization method')
    # optimization
    parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
#     parser.add_argument('--gradclip', type=float, default=grad_max_norm, help='gradient norm clip')
    parser.add_argument('--epochs', type=int, default=num_epochs, help='number of training epochs')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='directory to save the best model')
    parser.add_argument('--save_name', type=str, default=save_name, help='file name to save the best model')
    args = parser.parse_args()
    return args

args = parse_args()

########## logging setup
log_file = os.path.splitext(args.save_name)[0] + '.log'
f_log = open(os.path.join(args.save_dir, log_file), args.logmode)
def log(s):
    logging(s, f_log=f_log, print_=True, log_=True)
log('python ' + ' '.join(sys.argv))

log('-' * 30)
log(time.ctime())

########## load the dataset
log('loading dataset...')
data_train_path = [os.path.join(args.data_dir, name) for name in args.data_train]
data_val_path = [os.path.join(args.data_dir, args.data_val)]
data_test_path = [os.path.join(args.data_dir, args.data_test)]

train_ds = DstGraphDatasetLarge(data_train_path, time_window=args.tw, in_memory=args.in_mem)
train_loader = DstGraphLoader(train_ds, batch_size=args.bsz)

val_ds = DstGraphDatasetLarge(data_val_path, time_window=args.tw, in_memory=args.in_mem)
val_loader = DstGraphLoader(val_ds, batch_size=1)        # for correct calculation of evaluation metrics

test_ds = DstGraphDatasetLarge(data_test_path, time_window=args.tw, in_memory=args.in_mem)
test_loader = DstGraphLoader(test_ds, batch_size=1)      # for correct calculation of evaluation metrics


########## define the model, optimizer, and loss
model = DstGCNModel(args.in_channels,
                    args.enc_sizes,
                    args.n_classes,
                    non_linear='relu',
                    residual_hop=args.residual_hop,
                    pred_on='node',
                    edge_gate=None if args.edge_gate == 'None' else args.edge_gate)
log('model ' + '-' * 10)
log(repr(model))
# print('Initially:')
# print_cuda_mem()
if args.devid > -1:
    model.to(f'cuda:{args.devid}')
# print('After loading model to cuda device:')
# print_cuda_mem()

criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(alpha=1, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)
early_stopper = EarlyStopping(patience=5, verbose=True)

########## train the model

def test(model, data_loader, devid=args.devid):
    model.eval()
    loss_avg = 0
    acc = fpr = fnr = rec = prc = 0
    with torch.no_grad():
        for n, batch in enumerate(data_loader):        # here batch size is 1 for val and test data
            if devid > -1:
                batch.to(f'cuda:{devid}')
            x = model(batch.x, batch.edge_index_list, norm_method=args.norm_method)
            y = batch.y.view(-1).long()
            loss = criterion(x, y)
            loss_avg += float(loss)
            pred = x.argmax(dim=1)
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

best_epoch = 0
start = time.time()
# best_loss = float('inf')
for ep in range(args.epochs):
    loss_avg_train = 0
    num_train_graph = 0
    model.train()
    for batch in train_loader:
        if args.devid > -1:
            batch.to(f'cuda:{args.devid}')
#         print('After loading data to cuda device:')
#         print_cuda_mem()
        optimizer.zero_grad()
        x = model(batch.x, batch.edge_index_list, norm_method=args.norm_method)
        loss = criterion(x, batch.y.view(-1).long())
#         print('After a forward pass:')
#         print_cuda_mem()
        loss_avg_train += float(loss)
        num_train_graph += args.bsz
#        breakpoint()
        loss.backward()
        optimizer.step()
        
        if num_train_graph % 60 == 0:
            with torch.no_grad():
                pred = x.argmax(dim=1)
                y = batch.y.view(-1).long()
                acc = accuracy(pred, y)
                fpr = false_positive_rate(pred, y)
                fnr = false_negative_rate(pred, y)
                rec = recall(pred, y)
                prc = precision(pred, y)
            log(f'epoch: {ep + 1}, passed number of st-graphs: {num_train_graph} ({args.tw * num_train_graph} total graphs), train running loss: {loss_avg_train / num_train_graph:.5f} (passed time: {timeSince(start)})')
            log(f'                 false positive rate: {fpr:.6f}, false negative rate: {fnr:.5f}, recall: {rec:.5f}, precision: {prc:.5f}')
#             f_log.flush()
#             os.fsync(f_log.fileno())
        
    loss_avg, acc, fpr, fnr, rec, prc = test(model, val_loader)
    log(f'Validation --- epoch: {ep + 1}, loss: {loss_avg:.5f}, acc: {acc:.5f}, false positive rate: {fpr:.6f}, false negative rate: {fnr:.5f}')
    log(f'                                recall (true positive rate): {rec:.5f}, precision: {prc:.5f}')
    scheduler.step(loss_avg)

    early_stopper(loss_avg)
    if early_stopper.improved:
        torch.save(model, os.path.join(args.save_dir, args.save_name))
        log(f'model saved at {os.path.join(args.save_dir, args.save_name)}.')
        best_epoch = ep
    elif early_stopper.early_stop:
        log(f'Early stopping here.')
        break
    else:
        pass
        
#     if loss_avg < best_loss:
#         best_loss = loss_avg
#         torch.save(model, os.path.join(args.save_dir, args.save_name))
#         log(f'model saved at {os.path.join(args.save_dir, args.save_name)}.')

    f_log.flush()
    os.fsync(f_log.fileno())
  
if early_stopper.improved:
    best_model = model
else:
    best_model = torch.load(os.path.join(args.save_dir, args.save_name))
loss_avg, acc, fpr, fnr, rec, prc = test(best_model, test_loader)
log('*' * 12 + f' best model obtained after epoch {best_epoch + 1}, saved at {os.path.join(args.save_dir, args.save_name)} ' + '*' * 12)
log(f'Testing --- loss: {loss_avg:.5f}, acc: {acc:.5f}, false positive rate: {fpr:.6f}, false negative rate: {fnr:.5f}')
log(f'            recall (true positive rate): {rec:.5f}, precision: {prc:.5f}')

