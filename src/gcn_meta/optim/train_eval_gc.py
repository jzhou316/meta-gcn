import torch


def train(model, optimizer, data_loader, criterion, device, *, log_interval=None, logger=None):
    """
    Training for one epoch for graph classification task.
    Loss should be calculated as the mean loss by default in `criterion`.
    """
    model.train()

    loss_total = 0
    num_graphs = 0
    for n, batch in enumerate(data_loader, start=1):
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        loss_total += float(loss) * batch.num_graphs
        num_graphs += batch.num_graphs

        if log_interval is not None and (num_graphs % log_interval == 0 or n == len(data_loader)):
            logging = print if logger is None else logger.info
            logging('----- ' + f'passed number of graphs: {num_graphs}, training loss: {loss_total / num_graphs: .5f}')

    return loss_total / num_graphs


def eval(model, data_loader, criterion, device):
    """
    Evaluation for graph classification task.
    """
    model.eval()
    loss_total = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss_total += float(loss) * batch.num_graphs

            correct += out.argmax(dim=1).eq(batch.y).sum().item()

    return loss_total / len(data_loader.dataset), correct / len(data_loader.dataset)
