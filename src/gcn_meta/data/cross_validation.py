from sklearn.model_selection import StratifiedKFold
import torch


def k_fold_gc(dataset, folds):
    """
    K-fold data split for graph classification dataset.
    Adapted from https://github.com/anonymousOPT/OTCoarsening/blob/master/src/train_eval_opt.py.
    """
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices
