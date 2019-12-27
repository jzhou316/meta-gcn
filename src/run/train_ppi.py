import argparse
import os
import os.path as osp
import random
import time

import torch
import torch.nn as nn
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import f1_score

import sys

sys.path.insert(0, '../')
from gcn_meta.data.dataset import MyPPI
from gcn_meta.models.gcn_model import GCNModel
from gcn_meta.optim.earlystop import EarlyStopping
from gcn_meta.optim.focal_loss import FocalLoss
from gcn_meta.optim.misc import timeSince, logging, print_cuda_mem

########## some default parameters ##########
dataset = 'PPI'       # not used
add_sl = False

save_dir = '../saved_models/' + dataset.lower()
save_name = 'temp_model.pt'

batch_size = 1  # not used for citation dataset since only one graph

devid = 0
seed = 0

in_channels = 1  # 50 for PPI (taken care of automatically)
num_classes = 2  # 121 for PPI (taken care of automatically)

enc_sizes = [32, 32, 32, 32, 32, 32, 32, 32]
dropout = 0.5
residual_hop = 0
nodemodel = 'additive'  # 'attention', 'hardattention'
final = 'none'  # 'proj'

nheads = [1]  # number of heads in multihead attention
att_act = 'lrelu'  # 'none'
att_dropout = 0
att_dir = 'in'
att_combine = 'cat'
temperature = 0.1  # for hard attention using Gumbel-Softmax trick
sample = False  # whether to do sampling at testing time for hard attention

deg_norm = 'sm'
aggr = 'add'
edge_gate = 'None'
bias = False

# if_focal_loss = False
learning_rate = 0.001
# grad_max_norm = 5

num_epochs = 5


#############################################


def parse_args():
    parser = argparse.ArgumentParser(description='Training a GCN model.')
    # general setting
    parser.add_argument('--devid', type=int, default=devid, help='device id; -1 for CPU')
    parser.add_argument('--seed', type=int, default=seed, help='random seed')
    parser.add_argument('--logmode', type=str, default='w', help='logging file mode')
    # data loading
    parser.add_argument('--dataset', type=str, default=dataset, choices=['PPI'],
                        help='dataset name')
    parser.add_argument('--add_sl', type=int, default=add_sl, help='whether to add self loops')
    # parser.add_argument('--data_dir', type=str, default=data_dir, help='directory to find the dataset')
    # parser.add_argument('--data_train', type=str, default=data_train, help='file name of the training dataset')
    # parser.add_argument('--data_val', type=str, default=data_val, help='file name of the validation dataset')
    # parser.add_argument('--data_test', type=str, default=data_test, help='file name of the test dataset')
    parser.add_argument('--bsz', type=int, default=batch_size, help='batch size')
    # parser.add_argument('--in_mem', action='store_true', help='whether to load all the data into memory')
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

args.save_dir = '../saved_models/' + args.dataset.lower()
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

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
# log('\n')
# log(args)
log('-' * 30)
log(time.ctime())

########## load the dataset
log('loading dataset...')
args.dataset = 'PPI'

path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', args.dataset)
train_dataset = MyPPI(path, split='train', add_sl=bool(args.add_sl))
val_dataset = MyPPI(path, split='val', add_sl=bool(args.add_sl))
test_dataset = MyPPI(path, split='test', add_sl=bool(args.add_sl))

# train_dataset = PPI(path, split='train')
# val_dataset = PPI(path, split='val')
# test_dataset = PPI(path, split='test')
#
# log('adding self-loops...')
# for g in train_dataset:
#     g.edge_index, _ = add_remaining_self_loops(g.edge_index)
# for g in val_dataset:
#     g.edge_index, _ = add_remaining_self_loops(g.edge_index)
# for g in test_dataset:
#     g.edge_index, _ = add_remaining_self_loops(g.edge_index)

train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

########## define the model, optimizer, and loss
args.in_channels = train_dataset.num_features
args.n_classes = train_dataset.num_classes

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

# if not args.focal:
#     criterion = nn.CrossEntropyLoss()
# else:
#     criterion = FocalLoss(alpha=1, gamma=2)
criterion = nn.BCEWithLogitsLoss()        # 121 multi-classes

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)
early_stopper = EarlyStopping(patience=5, verbose=True)


########## train the model

def test(model, data_loader, criterion, device=device):
    model.eval()
    loss_avg = 0
    f1 = 0
    with torch.no_grad():
        for batch in data_loader:    # here batch size is 1 for val and test data
            batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y)
            loss_avg += float(loss)
            pred = (logits > 0).float()
            f1 += f1_score(batch.y.cpu().numpy(), pred.cpu().numpy(), average='micro') if pred.sum() > 0 else 0
    loss_avg = loss_avg / len(data_loader)
    f1 = f1 / len(data_loader)
    return loss_avg, f1


best_epoch = 0
start = time.time()
# best_loss = float('inf')
for ep in range(args.epochs):
    model.train()

    loss_avg_train = 0
    f1_avg_train = 0
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y)
        loss_avg_train += loss.float()
        loss.backward()
        optimizer.step()
        pred = (logits > 0).float()
        f1_avg_train += f1_score(batch.y.cpu().numpy(), pred.cpu().numpy(), average='micro') if pred.sum() > 0 else 0
    loss_avg_train = loss_avg_train / len(train_loader)
    f1_avg_train = f1_avg_train / len(train_loader)

    val_loss, val_f1 = test(model, val_loader, criterion, device)
    test_loss, test_f1 = test(model, test_loader, criterion, device)

    log(f'epoch: {ep + 1}, loss: train: {loss_avg_train:.5f}, val: {val_loss:.5f}, test: {test_loss:.5f}')
    log(f'          f1: train: {f1_avg_train:.4f}, val: {val_f1:.4f}, test: {test_f1:.4f} '
        f'(passed time: {timeSince(start)})')

    # scheduler.step(val_loss)

    early_stopper(val_loss)
    #     early_stopper(-val_f1)
    if early_stopper.improved:
        torch.save(model, os.path.join(args.save_dir, args.save_name))
        log(f'model saved at {os.path.join(args.save_dir, args.save_name)}.')
        best_epoch = ep
    elif early_stopper.early_stop:
        log(f'Early stopping here.')
    #         break
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
test_loss, test_f1 = test(model, test_loader, criterion, device)
log('*' * 12 + f' best model obtained after epoch {best_epoch + 1},'
               f'saved at {os.path.join(args.save_dir, args.save_name)} ' + '*' * 12)
log(f'Testing --- loss: {test_loss:.5f}, micro f1: {test_f1:.4f}')
