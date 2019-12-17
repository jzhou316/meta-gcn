from itertools import product

import random
import argparse
import os
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

import torch
import numpy as np

from gcn import GCN, GCNWithJK
from graph_sage import GraphSAGE, GraphSAGEWithJK
from gin import GIN0, GIN0WithJK, GIN, GINWithJK
from graclus import Graclus
from top_k import TopK
from sag_pool import SAGPool
from diff_pool import DiffPool
from edge_pool import EdgePool
from global_attention import GlobalAttentionNet
from set2set import Set2SetNet
from sort_pool import SortPool
from hard_pool import HardPool

import sys
sys.path.insert(0, '../')
from gcn_meta.optim.utils import logging_config

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--random_state', type=int, default=12345, help='random state for k-fold data split')
parser.add_argument('--repeat', type=int, default=1, help='number of repeat runs with the same data split')
parser.add_argument('--add_sl', type=int, default=0, help='whether to add self loops')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_dir', type=str, default=None, help='directory to save results')
parser.add_argument('--log_details', type=int, default=0, help='Whether to log details for every epoch in every fold')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--es_patience', type=int, default=-1, help='patience number for early stopping')
# do not use early stopping now, as there is some issue and seems early stopping doesn't do much
args = parser.parse_args()

layers = [1, 2, 3, 4, 5]
hiddens = [16, 32, 64, 128]
datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']  # , 'COLLAB']
nets = [
    GCNWithJK,
    GraphSAGEWithJK,
    GIN0WithJK,
    GINWithJK,
    Graclus,
    TopK,
    SAGPool,
    DiffPool,
    EdgePool,
    GCN,
    GraphSAGE,
    GIN0,
    GIN,
    GlobalAttentionNet,
    Set2SetNet,
    SortPool,
    HardPool,
]

datasets = ['PROTEINS']
#datasets = ['MUTAG']
#nets = [SAGPool]
#nets = [EdgePool]
nets = [HardPool]
#nets = [TopK]

layers = [4]
hiddens = [64]


if args.save_dir is None:
    args.save_dir = os.path.join('../saved_benchmark_gc', datasets[0].lower())
os.makedirs(args.save_dir, exist_ok=True)

log_name = nets[0].__name__ + '-' + str(layers[0]) + '-' + str(hiddens[0])

logger = logging_config(__name__, folder=args.save_dir, name=log_name, filemode='w')


def logger_cv(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    train_loss, val_loss, test_acc = info['train_loss'], info['val_loss'], info['test_acc']
    logger.info('{:02d}/{:03d}: Train Loss: {:.4f}, Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, train_loss, val_loss, test_acc))


random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    logger.info('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for num_layers, hidden in product(layers, hiddens):
        logger.info(f'num_layers: {num_layers}, hidden: {hidden}')
        dataset = get_dataset(dataset_name,
                              sparse=Net != DiffPool,
                              x_deg=True,
                              add_sl=args.add_sl,
                              cleaned=False)
        model = Net(dataset, num_layers, hidden)

        # repeat runs, as the GCN models have randomness themselves
        rpt_losses, rpt_accs, rpt_stds = [], [], []
        rpt_best = (float('inf'), 0, 0)
        for rpt in range(args.repeat):
            loss, acc, std = cross_validation_with_val_set(
                dataset,
                model,
                folds=10,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                random_state=args.random_state,
                es_patience=args.es_patience,
                logger=logger,
                log_details=bool(args.log_details),
            )
            rpt_losses.append(loss)
            rpt_accs.append(acc)
            rpt_stds.append(std)
            # if loss < rpt_best[0]:
            #     rpt_best = (loss, acc, std)
            #     rpt_best_id = rpt
            #     # TODO: save the best model here

        loss = sum(rpt_losses) / args.repeat
        acc = sum(rpt_accs) / args.repeat
        std = sum(rpt_stds) / args.repeat
        logger.info(f'Average - val_loss: {loss:.4f}, test_acc: {acc:.3f} ± {std:.3f}')

        rpt_best_id = np.argmin(rpt_losses)
        # rpt_best_id = np.argmax(rpt_accs)
        loss, acc, std = rpt_losses[rpt_best_id], rpt_accs[rpt_best_id], rpt_stds[rpt_best_id]
        logger.info(f'Best - val_loss: {loss:.4f}, test_acc: {acc:.3f} ± {std:.3f}')

        if loss < best_result[0]:
            best_result = (loss, acc, std)
            best_hyper = (num_layers, hidden)

    desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    logger.info('Best result - {}'.format(desc))
    logger.info('Best hyper-parameter - {}, {}'.format(*best_hyper))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
logger.info('-----\n{}'.format('\n'.join(results)))
