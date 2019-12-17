import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

import sys
sys.path.insert(0, '../src')
from gcn_meta.optim.earlystop import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, random_state=12345, es_patience=-1,
                                  logger=None, log_details=False):

    logging = logger.info if logger is not None else print

    val_losses, val_accs, test_losses, test_accs, durations = [], [], [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds, random_state))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopper = EarlyStopping(patience=es_patience, mode='min', verbose=True, logger=None)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)

            val_loss, val_acc = eval_loss_acc(model, val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            test_loss, test_acc = eval_loss_acc(model, test_loader)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            if log_details:
                logging(f'Fold {(fold + 1):02d} / Epoch {(epoch + 1):03d}: Train loss: {train_loss:.4f}, '
                        f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.3f}, '
                        f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.3f}')

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

            early_stopper(val_loss)
            if early_stopper.early_stop:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    val_loss, val_acc, duration = tensor(val_losses), tensor(val_accs), tensor(durations)
    test_loss, test_acc = tensor(test_losses), tensor(test_accs)
    val_loss, val_acc = val_loss.view(folds, epochs), val_acc.view(folds, epochs)
    test_loss, test_acc = test_loss.view(folds, epochs), test_acc.view(folds, epochs)

    # val_loss, argmin = val_loss.min(dim=1)
    # val_acc = val_acc[torch.arange(folds, dtype=torch.long), argmin]

    val_acc, argmin = val_acc.max(dim=1)
    val_loss = val_loss[torch.arange(folds, dtype=torch.long), argmin]

    test_loss = test_loss[torch.arange(folds, dtype=torch.long), argmin]
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), argmin]

    val_loss_mean = val_loss.mean().item()
    val_acc_mean = val_acc.mean().item()
    val_acc_std = val_acc.std().item()
    duration_mean = duration.mean().item()
    test_loss_mean = test_loss.mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    logging('Best epoch for each fold:', argmin.cpu().numpy().tolist())
    logging('Val Loss: {:.4f}, Val Accuracy: {:.3f} ± {:.3f}, '
            'Test Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
            format(val_loss_mean, val_acc_mean, val_acc_std,
                   test_loss_mean, test_acc_mean, test_acc_std, duration_mean))

    return val_loss_mean, val_acc_mean, val_acc_std, test_loss_mean, test_acc_mean, test_acc_std


def k_fold(dataset, folds, random_state=12345):
    skf = StratifiedKFold(folds, shuffle=True, random_state=random_state)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        # print(out.sum(dim=0).detach().cpu().numpy())
        # print(loss)
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            # print(out.sum(dim=0).detach().cpu().numpy())
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def eval_loss_acc(model, loader):
    model.eval()

    loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)
