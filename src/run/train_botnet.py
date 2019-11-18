import argparse
import os
import random
import time

import torch
import torch.nn as nn

import sys

sys.path.insert(0, '../')
from gcn_meta.data.dataset import GraphDataset
from gcn_meta.data.dataloader import GraphDataLoader
from gcn_meta.models.gcn_model import GCNModel
from gcn_meta.optim.earlystop import EarlyStopping
from gcn_meta.optim.focal_loss import FocalLoss
from gcn_meta.optim.metrics import *
from gcn_meta.optim.misc import timeSince, logging, print_cuda_mem


##### some default parameters
# data_dir = '/n/holylfs/LABS/tata_yu_rush_level1_data/TataProject/data_botnet'
data_dir = '../../data/botnet/ev10k'
data_train = 'chord_no100k_ev10k_us_train.hdf5'
data_val = 'chord_no100k_ev10k_us_val.hdf5'
data_test = 'chord_no100k_ev10k_us_test.hdf5'
# data_train_path = os.path.join(data_dir, data_train)
# data_val_path = os.path.join(data_dir, data_val)
# data_test_path = os.path.join(data_dir, data_test)
save_dir = '../saved_models/botnet_ev10k'
save_name = 'temp_model.pt'

## quick testing
# data_train = 'chord_no100k_ev10k_us_small.hdf5'
# data_val = 'chord_no100k_ev10k_us_small.hdf5'
# data_test = 'chord_no100k_ev10k_us_small.hdf5'
## quick testing on Azure
# data_dir = '/media/work/TataProject/data_botnet'
# data_train = 'chord_no100k_ev10k_us_small.hdf5'
# data_val = 'chord_no100k_ev10k_us_small.hdf5'
# data_test = 'chord_no100k_ev10k_us_small.hdf5'
# save_dir = './saved_models'
# save_name = 'temp_model.pt'

in_memory = False
batch_size = 1

devid = 0
seed = 0

in_channels = 1
enc_sizes = [32, 32, 32, 32, 32, 32, 32, 32]
residual_hop = 0
edge_gate = 'None'
num_classes = 2
nodemodel = 'additive'  # 'attention', 'hardattention'
final = 'proj'  # 'none'

nheads = 1  # number of heads in multihead attention
att_act = 'none'
att_dropout = 0
att_dir = 'in'
att_combine = 'cat'
temperature = 0.1  # for hard attention using Gumbel-Softmax trick
sample = 0

deg_norm = 'sm'
aggr = 'add'
dropout = 0
bias = False

# if_focal_loss = False
learning_rate = 0.001
# grad_max_norm = 5

num_epochs = 5


#####

def parse_args():
    parser = argparse.ArgumentParser(description='Training a GCN model.')
    # general setting
    parser.add_argument('--devid', type=int, default=devid, help='device id; -1 for CPU')
    parser.add_argument('--seed', type=int, default=seed, help='random seed')
    parser.add_argument('--logmode', type=str, default='w', help='logging file mode')
    # data loading
    parser.add_argument('--data_dir', type=str, default=data_dir, help='directory to find the dataset')
    parser.add_argument('--data_train', type=str, default=data_train, help='file name of the training dataset')
    parser.add_argument('--data_val', type=str, default=data_val, help='file name of the validation dataset')
    parser.add_argument('--data_test', type=str, default=data_test, help='file name of the test dataset')
    parser.add_argument('--bsz', type=int, default=batch_size, help='batch size')
    parser.add_argument('--in_mem', type=int, default=in_memory, help='whether to load all the data into memory')
    # model
    parser.add_argument('--in_channels', type=int, default=in_channels, help='input node feature size')
    parser.add_argument('--enc_sizes', type=int, nargs='*', default=enc_sizes, help='encoding node feature sizes')
    parser.add_argument('--act', type=str, default='relu', help='non-linear activation function after adding residual')
    parser.add_argument('--layer_act', type=str, default='relu', help='non-linear activation function for each layer')
    parser.add_argument('--residual_hop', type=int, default=residual_hop, help='residual per # layers')
    parser.add_argument('--edge_gate', type=str, choices=['None', 'proj', 'free'], default=edge_gate,
                        help='types of edge gate')
    parser.add_argument('--n_classes', type=int, default=num_classes, help='number of classes for the output layer')
    parser.add_argument('--nodemodel', type=str, default=nodemodel, choices=['additive', 'attention', 'hardattention'],
                        help='name of node model class')
    parser.add_argument('--final', type=str, default=final, choices=['none', 'proj'], help='final output layer')
    # attention (temperature is only for hard attention)
    parser.add_argument('--nheads', type=int, nargs='*', default=nheads, help='number of heads in multihead attention')
    parser.add_argument('--att_act', type=str, default=att_act, choices=['none', 'lrelu', 'relu'],
                        help='attention activation function in multihead attention')
    parser.add_argument('--att_dropout', type=float, default=att_dropout,
                        help='attention dropout in multihead attention')
    parser.add_argument('--att_dir', type=str, default=att_dir, help='attention direction in multihead attention')
    parser.add_argument('--att_combine', type=str, default=att_combine, choices=['cat', 'add', 'mean'],
                        help='multihead combination method in multihead attention')
    parser.add_argument('--att_combine_out', type=str, default=att_combine, choices=['cat', 'add', 'mean'],
                        help='multihead combination method in multihead attention for the last output attention layer')
    parser.add_argument('--temperature', type=float, default=temperature,
                        help='temperature in multihead HARD attention')
    parser.add_argument('--sample', type=int, default=sample, help='whether to sample at testing for HARD attention')
    # other model arguments
    parser.add_argument('--deg_norm', type=str, choices=['None', 'sm', 'rw'], default=deg_norm,
                        help='degree normalization method')
    parser.add_argument('--aggr', type=str, choices=['add', 'mean', 'max'], default=aggr,
                        help='feature aggregation method')
    parser.add_argument('--dropout', type=float, default=dropout, help='dropout probability')
    parser.add_argument('--bias', type=int, default=bias, help='whether to include bias in the model')
    # optimization
    parser.add_argument('--focal', action='store_true', help='whether to use focal loss or normal CrossEntropyLoss')
    parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 penalty)')
    # parser.add_argument('--gradclip', type=float, default=grad_max_norm, help='gradient norm clip')
    parser.add_argument('--epochs', type=int, default=num_epochs, help='number of training epochs')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='directory to save the best model')
    parser.add_argument('--save_name', type=str, default=save_name, help='file name to save the best model')
    args = parser.parse_args()
    return args


args = parse_args()

os.makedirs(args.save_dir, exist_ok=True)

########## random seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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
data_train_path = os.path.join(args.data_dir, args.data_train)
data_val_path = os.path.join(args.data_dir, args.data_val)
data_test_path = os.path.join(args.data_dir, args.data_test)

train_ds = GraphDataset(data_train_path, in_memory=args.in_mem)
train_loader = GraphDataLoader(train_ds, batch_size=args.bsz, num_workers=0)

val_ds = GraphDataset(data_val_path, in_memory=args.in_mem)
val_loader = GraphDataLoader(val_ds, batch_size=1, num_workers=0)  # for correct calculation of evaluation metrics

test_ds = GraphDataset(data_test_path, in_memory=args.in_mem)
test_loader = GraphDataLoader(test_ds, batch_size=1, num_workers=0)  # for correct calculation of evaluation metrics

########## define the model, optimizer, and loss
if len(args.nheads) < len(args.enc_sizes):
    assert len(args.nheads) == 1
    args.nheads = args.nheads * len(args.enc_sizes)
elif len(args.nheads) == len(args.enc_sizes):
    pass
else:
    raise ValueError

final_layer_config = {'att_combine': args.att_combine_out}
model = GCNModel(args.in_channels,
                 args.enc_sizes,
                 args.n_classes,
                 non_linear=args.act,
                 non_linear_layer_wise=args.layer_act,
                 residual_hop=args.residual_hop,
                 dropout=args.dropout,
                 final_type=args.final,
                 pred_on='node',
                 nodemodel=args.nodemodel,
                 deg_norm=None if args.deg_norm == 'None' else args.deg_norm,
                 edge_gate=None if args.edge_gate == 'None' else args.edge_gate,
                 aggr=args.aggr,
                 bias=bool(args.bias),
                 nheads=args.nheads,
                 att_act=args.att_act,
                 att_dropout=args.att_dropout,
                 att_dir=args.att_dir,
                 att_combine=args.att_combine,
                 temperature=args.temperature,
                 sample=bool(args.sample),
                 final_layer_config=final_layer_config,
                 )
log('model ' + '-' * 10)
log(repr(model))
# print('Initially:')
# print_cuda_mem()
# if args.devid > -1:
#     model.to(f'cuda:{args.devid}')
device = f'cuda:{args.devid}' if args.devid > -1 else 'cpu'
model.to(device)
# print('After loading model to cuda device:')
# print_cuda_mem()

if not args.focal:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = FocalLoss(alpha=1, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)
early_stopper = EarlyStopping(patience=5, verbose=True)


########## train the model

def test(model, data_loader, criterion, device=device):
    model.eval()
    loss_avg = 0
    acc = fpr = fnr = rec = prc = 0
    with torch.no_grad():
        for n, batch in enumerate(data_loader):  # here batch size is 1 for val and test data
            batch.to(device)
            if n == 0:    # store the hard attention on the first test graph
                attn = []
                x = model(batch.x[:, 0].view(-1, 1), batch.edge_index, deg_K=batch.x[:, 1], attn_store=attn)
                attn_store_path = os.path.join(args.save_dir, os.path.splitext(args.save_name)[0] + '_test0-attn.pt')
                torch.save(attn, attn_store_path)
                log(f'Attention scores (of the first graph in the test set) saved at {attn_store_path}')
            else:
                x = model(batch.x[:, 0].view(-1, 1), batch.edge_index, deg_K=batch.x[:, 1])
            y = batch.y.long()
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
    model.train()

    loss_avg_train = 0
    num_train_graph = 0
    for batch in train_loader:
        batch.to(device)
        #         print('After loading data to cuda device:')
        #         print_cuda_mem()
        optimizer.zero_grad()
        breakpoint()
        x = model(batch.x[:, 0].view(-1, 1), batch.edge_index, deg_K=batch.x[:, 1])
        loss = criterion(x, batch.y.long())
        #         print('After a forward pass:')
        #         print_cuda_mem()
        loss_avg_train += float(loss)
        num_train_graph += args.bsz
        #        breakpoint()
        loss.backward()
        optimizer.step()

        if num_train_graph % 96 == 0:
            with torch.no_grad():
                pred = x.argmax(dim=1)
                y = batch.y.long()
                acc = accuracy(pred, y)
                fpr = false_positive_rate(pred, y)
                fnr = false_negative_rate(pred, y)
                rec = recall(pred, y)
                prc = precision(pred, y)
            log(
                f'epoch: {ep + 1}, passed number of graphs: {num_train_graph}, train running loss: {loss_avg_train / num_train_graph:.5f} (passed time: {timeSince(start)})')
            log(
                f'                 false positive rate: {fpr:.6f}, false negative rate: {fnr:.5f}, recall: {rec:.5f}, precision: {prc:.5f}')
    #             f_log.flush()
    #             os.fsync(f_log.fileno())

    loss_avg, acc, fpr, fnr, rec, prc = test(model, val_loader)
    log(
        f'Validation --- epoch: {ep + 1}, loss: {loss_avg:.5f}, acc: {acc:.5f}, false positive rate: {fpr:.6f}, false negative rate: {fnr:.5f}')
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
log(
    '*' * 12 + f' best model obtained after epoch {best_epoch + 1}, saved at {os.path.join(args.save_dir, args.save_name)} ' + '*' * 12)
log(f'Testing --- loss: {loss_avg:.5f}, acc: {acc:.5f}, false positive rate: {fpr:.6f}, false negative rate: {fnr:.5f}')
log(f'            recall (true positive rate): {rec:.5f}, precision: {prc:.5f}')
