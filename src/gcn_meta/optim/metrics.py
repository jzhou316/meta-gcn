"""
Evaluation metrics for binary classification.

`pred` and `target` are both `torch.LongTensor`s of the same length.
"""


def accuracy(pred, target):
    return (pred == target).sum().item() / target.numel()


def true_positive(pred, target):
    return (target[pred == 1] == 1).sum().item()


def false_positive(pred, target):
    return (target[pred == 1] == 0).sum().item()


def true_negative(pred, target):
    return (target[pred == 0] == 0).sum().item()


def false_negative(pred, target):
    return (target[pred == 0] == 1).sum().item()


def recall(pred, target):
    """
    Or true positive rate.
    """
    return true_positive(pred, target) / (target == 1).sum().item()


def precision(pred, target):
    try:
        prec = true_positive(pred, target) / (pred == 1).sum().item()
        return prec
    except:  # divide by zero
        return -1


def f1_score(pred, target):
    prec = precision(pred, target)
    rec = recall(pred, target)
    return 2 * (prec * rec) / (prec + rec)


def false_positive_rate(pred, target):
    return false_positive(pred, target) / (target == 0).sum().item()


def false_negative_rate(pred, target):
    """
    Or 1 - recall/true_positive_rate
    """
    return false_negative(pred, target) / (target == 1).sum().item()
