import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

sys.path.insert(0, '../')
from gcn_meta.optim.utils import logging_config
from gcn_meta.data.dataset import MyTUDataset
from gcn_meta.optim.earlystop import EarlyStopping
from gcn_meta.data.cross_validation import k_fold_gc


# ========== some default parameters ==========
devid = 0
seed = 0

data_dir = '../../data'
data_name = 'DD'    # 'DD', 'ENZYMES', 'PROTEINS', 'COLLAB', 'NCI1', 'NCI109', 'MUTAG', 'IMDB-BINARY', etc.
# from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
add_sl = False
batch_size = 1
save_dir = os.path.join('../saved_models_gc', data_dir.lower())
save_name = 'temp_model.pt'

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
    parser.add_argument('--log_interval', type=int, default=80, help='logging interval during training')
    # data loading
    parser.add_argument('--data_dir', type=str, default=data_dir, help='data directory')
    parser.add_argument('--data_name', type=str, default=dataset, choices=['DD', 'ENZYMES', 'PROTEINS', 'COLLAB',
                                                                           'NCI1', 'NCI109', 'MUTAG', 'IMDB-BINARY'],
                        help='dataset name')
    parser.add_argument('--add_sl', type=int, default=add_sl, help='whether to add self loops')
    parser.add_argument('--folds', type=int, default=10, help='number of fold for cross validation')
    parser.add_argument('--bsz', type=int, default=batch_size, help='batch size')
    # model

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
dataset = MyTUDataset(os.path.join(args.data_dir, args.data_name), args.data_name, x_deg=True, add_sl=args.add_sl)


# ======== define the model, optimizer, and loss
# model =

logger.info('model ' + '-' * 10)
logger.info(repr(model))

# model.to(device)

criterion = nn.CrossEntropyLoss()


# ======== train the model: K-fold cross validation
for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold_gc(dataset, args.folds))):
    train_dataset = data.Subset(dataset, train_idx)
    test_dataset = data.Subset(dataset, test_idx)
    val_dataset = data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, args.bsz, shuffle=True)
    val_loader = DataLoader(val_dataset, args.bsz, shuffle=False)
    test_loader = DataLoader(test_dataset, args.bsz, shuffle=False)

    model.to(device).reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)
    early_stopper = EarlyStopping(patience=5, mode='min', verbose=True, logger=logger)

    breakpoint()

