import torch.nn as nn


class OutLayerIdentity(nn.Module):
    """
    Final output layer (an overhead) on top of GCN outputs on nodes:
    Identity.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class OutLayerProject(nn.Module):
    """
    Final output layer (an overhead) on top of GCN outputs on nodes:
    A linear projection.
    """

    def __init__(self, enc_size, num_classes):
        super().__init__()
        self.enc_size = enc_size
        self.num_classes = num_classes
        self.proj = nn.Linear(enc_size, num_classes)

    def forward(self, x):
        return self.proj(x)
