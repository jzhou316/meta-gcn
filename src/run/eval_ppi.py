import os.path as osp
import sys
import random

import torch
from torch_geometric.datasets import PPI
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score
sys.path.insert(0, '../')
from gcn_meta.models.gcn_model import GCNModel    # model source file for correct loading


def test(model, data_loader, criterion, device, attn_store=None):
    model.eval()
    loss_avg = 0
    f1 = 0
    with torch.no_grad():
        for n, batch in enumerate(data_loader):    # here batch size is 1 for val and test data
            batch.to(device)
            if n == 0:
                logits = model(batch.x, batch.edge_index, attn_store=attn_store)
            else:
                logits = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y)
            loss_avg += float(loss)
            pred = (logits > 0).float()
            f1 += f1_score(batch.y.cpu().numpy(), pred.cpu().numpy(), average='micro') if pred.sum() > 0 else 0
    loss_avg = loss_avg / len(data_loader)
    f1 = f1 / len(data_loader)
    return loss_avg, f1


if __name__ == '__main__':
    random.seed(0)

    # load dataset
    dataset = 'PPI'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', dataset)
    # train_dataset = PPI(path, split='train')
    # val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

    # add self-loops
    # for g in train_dataset:
    #     g.edge_index, _ = add_remaining_self_loops(g.edge_index)
    # for g in val_dataset:
    #     g.edge_index, _ = add_remaining_self_loops(g.edge_index)
    for g in test_dataset:
        g.edge_index, _ = add_remaining_self_loops(g.edge_index)

    # data loader
    # bsz = 1
    # train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # load model
    model_dir = '../saved_models/' + dataset.lower()
    model_name = 'temp_model.pt'
    model = torch.load(osp.join(model_dir, model_name), map_location='cpu')

    # test model
    device = torch.device('cuda:0')
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()  # 121 multi-classes
    attn_store = []
    loss_avg, f1 = test(model, test_loader, criterion, device, attn_store=attn_store)
    print(f'Testing --- loss: {loss_avg:.5f}, micro f1: {f1:.4f}')
    attn_store_path = osp.join(model_dir, osp.splitext(model_name)[0] + '_test0-attn.pt')
    torch.save(attn_store, attn_store_path)
    print(f'Attention scores (of the first graph in the test set) saved at {attn_store_path}')
