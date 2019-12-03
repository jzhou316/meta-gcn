import os
import argparse
import random
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.utils.data as data

sys.path.insert(0, '../')
from gcn_meta.optim.utils import logging_config, time_since
from gcn_meta.data.dataset import MyTUDataset
from gcn_meta.optim.earlystop import EarlyStopping
from gcn_meta.data.cross_validation import k_fold_gc
from gcn_meta.data.dataloader import GraphDataLoaderDataset
from gcn_meta.models.gpool_hard_attention import VotePoolModel
from gcn_meta.optim.train_eval_gc import train, eval


# ========== some default parameters ==========
devid = 0
seed = 0

data_dir = '../../data'
data_name = 'DD'  # 'DD', 'ENZYMES', 'PROTEINS', 'COLLAB', 'NCI1', 'NCI109', 'MUTAG', 'IMDB-BINARY', etc.
# from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
add_sl = False
batch_size = 1
save_dir = os.path.join('../saved_models_gc', data_name.lower())
save_name = 'temp_model.pt'

enc_sizes = [32, 32, 32, 32, 32, 32]
residual = True
act = 'relu'
aggr = 'add'
dropout = 0.0
bias = False
nheads = [1]
att_act = 'lrelu'
att_dropout = 0.0
att_combine = 'cat'
temperature = 0.1
sample = False
p_keep = -1
relabel = False

folds = 10
# if_focal_loss = False
learning_rate = 0.001
# grad_max_norm = 5
num_epochs = 5


# ==============================================


def parse_args():
    parser = argparse.ArgumentParser(description='Training a GCN model for graph classification.')
    # general setting
    parser.add_argument('--devid', type=int, default=devid, help='device id; -1 for CPU')
    parser.add_argument('--seed', type=int, default=seed, help='random seed')
    parser.add_argument('--logmode', type=str, default='w', help='logging file mode')
    parser.add_argument('--log_interval', type=int, default=200, help='logging interval during training')
    # data loading
    parser.add_argument('--data_dir', type=str, default=data_dir, help='data directory')
    parser.add_argument('--data_name', type=str, default=data_name, choices=['DD', 'ENZYMES', 'PROTEINS', 'COLLAB',
                                                                             'NCI1', 'NCI109', 'MUTAG', 'IMDB-BINARY'],
                        help='dataset name')
    parser.add_argument('--add_sl', type=int, default=add_sl, help='whether to add self loops')
    parser.add_argument('--folds', type=int, default=folds, help='number of fold for cross validation')
    parser.add_argument('--no_cv', type=int, default=0, help='do not use cross validation '
                                                             '(but data split is still determined by --folds')
    parser.add_argument('--bsz', type=int, default=batch_size, help='batch size')
    # model
    parser.add_argument('--enc_sizes', type=int, nargs='*', default=enc_sizes, help='encoding node feature sizes')
    parser.add_argument('--act', type=str, default=act, help='non-linear activation function after adding residual')
    parser.add_argument('--residual', type=int, default=residual, help='whether to use residual')
    # parser.add_argument('--n_classes', type=int, default=num_classes, help='number of classes for the output layer')
    parser.add_argument('--aggr', type=str, choices=['add', 'mean', 'max'], default=aggr,
                        help='feature aggregation method')
    parser.add_argument('--dropout', type=float, default=dropout, help='dropout probability')
    parser.add_argument('--bias', type=int, default=bias, help='whether to include bias in the model')
    # attention (temperature is only for hard attention)
    parser.add_argument('--nheads', type=int, nargs='*', default=nheads, help='number of heads in multihead attention')
    parser.add_argument('--att_act', type=str, default=att_act, choices=['none', 'lrelu', 'relu'],
                        help='attention activation function in multihead attention')
    parser.add_argument('--att_dropout', type=float, default=att_dropout,
                        help='attention dropout in multihead attention')
    parser.add_argument('--att_combine', type=str, default=att_combine, choices=['cat', 'add', 'mean'],
                        help='multihead combination method in multihead attention')
    parser.add_argument('--temperature', type=float, default=temperature,
                        help='temperature in multihead HARD attention')
    parser.add_argument('--sample', type=int, default=sample, help='whether to sample at testing for HARD attention')
    parser.add_argument('--p_keep', type=float, default=p_keep, help='attention probability threshold for pruning')
    parser.add_argument('--relabel', type=int, default=relabel, help='whether to relabel nodes when pruning')
    # optimization
    parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 penalty)')
    # parser.add_argument('--gradclip', type=float, default=grad_max_norm, help='gradient norm clip')
    parser.add_argument('--epochs', type=int, default=num_epochs, help='number of training epochs')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='directory to save the best model')
    parser.add_argument('--save_name', type=str, default=save_name, help='file name to save the best model')
    args = parser.parse_args()
    return args


args = parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
# alternatively
# os.makedirs(args.save_dir, exist_ok=True)


# ======== random seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device(f'cuda:{args.devid}') if args.devid > -1 else torch.device('cpu')

# ======== logging setup
log_name = os.path.splitext(args.save_name)[0]
logger = logging_config(__name__, folder=args.save_dir, name=log_name, filemode=args.logmode)
# logger = logging_config(os.path.basename(__file__), folder=args.save_dir, name=log_name, filemode=args.logmode)

logger.info('python ' + ' '.join(sys.argv))
logger.info('-' * 30)
logger.info(args)
logger.info('-' * 30)
logger.info(time.ctime())
logger.info('-' * 30)

# ======== load the dataset
dataset = MyTUDataset(os.path.join(args.data_dir, args.data_name), args.data_name, x_deg=True, add_sl=bool(args.add_sl))


# ======== define the model, optimizer, and loss
# need to care about batching graph level output
in_channels = dataset.num_node_features
num_classes = dataset.num_classes
model = VotePoolModel(in_channels,
                      args.enc_sizes,
                      num_classes,
                      non_linear=args.act,
                      residual=bool(args.residual),
                      dropout=args.dropout,
                      att_act=args.att_act,
                      att_dropout=args.att_dropout,
                      att_combine=args.att_combine,
                      bias=bool(args.bias),
                      aggr=args.aggr,
                      temperature=args.temperature,
                      sample=bool(args.sample),
                      p_keep=None if args.p_keep < 0 else args.p_keep,
                      relabel=bool(args.relabel)
                      )

logger.info('model ' + '-' * 10)
logger.info(repr(model))

model.to(device)

criterion = nn.CrossEntropyLoss()

# ======== train the model: K-fold cross validation
if args.folds < 3:
    # minimal value is 2, but we want train/val/test, so minimal should be 3
    raise ValueError('args.folds should be at least 3 (standard is 10)')


loss_cv, acc_cv = [], []
for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold_gc(dataset, args.folds))):
    # logging information
    logger.info('\n')
    logger.info('*' * 12 + f' cross validation {fold + 1}/{args.folds} ' + '*' * 12)
    logger.info('\n')
    save_path = os.path.join(args.save_dir, args.save_name)
    save_path, ext = os.path.splitext(save_path)
    save_path = save_path + f'_cv{fold + 1}' + ext

    # load split dataset
    train_dataset = data.Subset(dataset, train_idx.view(-1, 1))
    test_dataset = data.Subset(dataset, test_idx.view(-1, 1))
    val_dataset = data.Subset(dataset, val_idx.view(-1, 1))
    # need to expand dimension, otherwise error when indexing a single element
    # TypeError: iteration over a 0-d array Python
    # reason is *_idx[idx] has size torch.Size([])

    train_loader = GraphDataLoaderDataset(train_dataset, args.bsz, shuffle=True)
    val_loader = GraphDataLoaderDataset(val_dataset, args.bsz, shuffle=False)
    test_loader = GraphDataLoaderDataset(test_dataset, args.bsz, shuffle=False)

    # initialize model, optimizer, etc.
    model.to(device).reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)
    early_stopper = EarlyStopping(patience=5, mode='min', verbose=True, logger=logger)

    # training
    start = time.time()
    for ep in range(args.epochs):
        logger.info(f'Training --- epoch: {ep + 1}/{args.epochs}')

        model.train()
        breakpoint()
        loss_avg_train, remaining_nodes = train(model, optimizer, train_loader, criterion, device,
                                                log_interval=args.log_interval, logger=logger)
        breakpoint()
        loss_avg, acc, remaining_nodes = eval(model, val_loader, criterion, device)

        logger.info(f'Validation --- epoch: {ep + 1}/{args.epochs}, loss: {loss_avg:.5f}, acc: {acc:.5f} '
                    f'(passed time: {time_since(start)})')

        scheduler.step(loss_avg)

        early_stopper(loss_avg)
        if early_stopper.improved:
            torch.save(model, save_path)
            logger.info(f'model saved at {save_path}.')
            best_epoch = ep
        elif early_stopper.early_stop:
            logger.info(f'Early stopping here.')
            break
        else:
            pass

    if early_stopper.improved:
        best_model = model
    else:
        best_model = torch.load(save_path)
    loss_avg, acc, remaining_nodes = eval(best_model, test_loader, criterion, device)
    logger.info('*' * 12 + f' best model obtained after epoch {best_epoch + 1}, '
                           f'saved at {save_path} ' + '*' * 12)
    logger.info(f'Testing --- loss: {loss_avg:.5f}, acc: {acc:.5f}')

    # record results for the current fold
    loss_cv.append(loss_avg)
    acc_cv.append(acc)

    if args.no_cv:
        # do not use cross validation, so we force the loop to break after first data split
        break

logger.info('=' * 50)
logger.info(f'Average loss from cross validation {sum(loss_cv) / len(loss_cv):.5f} '
            f'(std: {torch.std(torch.tensor(loss_cv)).item():.5f})')
logger.info(f'Average accuracy from cross validation {sum(acc_cv) / len(acc_cv):.5f} '
            f'(std: {torch.std(torch.tensor(acc_cv)).item():.5f})')
