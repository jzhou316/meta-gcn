import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss that puts more weights on the hard examples to help combat extreme class imbalance problem.
    This loss criterion is used on the class scores before softmax.

    According to `"Focal Loss for Dense Object Detection <https://arxiv.org/pdf/1708.02002.pdf>"`,
    the loss is computed as follows:

    .. math::

         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
        - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
            output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
            ‘mean’: the sum of the output will be divided by the number of elements
            in the output, ‘sum’: the output will be summed. Default: ‘mean'.

    Input:
        - scores (torch.Tensor): classification scores before softmax, size (N, C), where
            C = number of classes.
        - target (torch.Tensor): true class of each example, size (N,), where each value is
            :math:`0 ≤ targets[i] ≤ C−1`.

    Output:
        - loss (torch.Tensor): scaler if `reduction` is not `'None'`, otherwise vector of size N.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        assert reduction in ['mean', 'sum', 'none']
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-8
        
    def forward(self, scores, target):
        # calculate class probabilities
        probs = F.softmax(scores, dim=1) + self.eps
        # take out target probabilities
        probs = probs[torch.arange(len(target)), target]
        # calculate focal loss
        weight = torch.pow(1 - probs, self.gamma)
        loss = -self.alpha * weight * torch.log(probs)
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:    # self.reduction == 'none'
            pass
        
        return loss
