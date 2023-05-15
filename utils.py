import torch
from regularizer import MMD, L1


def train_loop(dataloader, model, loss_fn, optimizer, device, **kwargs):
    model.train()
    for data in dataloader:
        # Compute prediction and loss
        ids = data['input_ids'].to(device)
        labels = data['label'].to(device)
        pred = model(ids)
        loss = loss_fn(pred, labels)
        # add regularization into the loss, if any
        if kwargs["regularization"] == "l1":
            l1 = L1(model.embedding.weight)
            loss = loss + kwargs["weight"] * l1.compute()
        elif kwargs["regularization"] == "wmmd":
            mmd = MMD(model.embedding.weight, kwargs["stopping_idx"], device)
            mmd_value = mmd.compute(ids, labels)
            loss = loss - kwargs["weight"] * mmd_value

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        print(f"loss for mini-batch: {loss:>7f}")
            

def test_loop(dataloader, model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            ids = data['input_ids'].to(device)
            labels = data['label'].to(device)
            pred = model(ids)

            predicted = torch.argmax(pred, dim = 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Accuracy of the network on the test dataset: {acc:4} %')
    return acc

def collate_fn(batch):
    input_ids = [_[0] for _ in batch]
    label = [_[1] for _ in batch]
    label = torch.tensor(label, dtype = torch.long)
    input_ids = torch.tensor(input_ids, dtype = torch.long)
    return {'label': label, 'input_ids': input_ids}

