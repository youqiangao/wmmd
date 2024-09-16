import numpy as np
import torch
import torch.nn.functional as F
from regularizers import Regularizer
from torch.optim.lr_scheduler import LinearLR 
from sklearn.model_selection import KFold
from typing import Tuple
import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    regularizer: Regularizer,
) -> None:
    model.train()
    for idx, (vocab_id, label, count) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(vocab_id)
        loss = F.cross_entropy(outputs, label) + regularizer(ids=vocab_id, counts=count, labels=label)
        loss.backward()
        optimizer.step()


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for vocab_id, label, _ in test_loader:
            outputs = model(vocab_id)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct / total

def train_test(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    regularizer: Regularizer,
    cfg: DictConfig,
    verbose: bool = False,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=cfg.epochs)

    for epoch in range(cfg.epochs):
        train(model, train_loader, optimizer, regularizer)
        lr_scheduler.step()
        if verbose:
            acc = test(model, test_loader)
            logger.info(f"Epoch: {epoch}, Accuracy: {acc}")

    return test(model, test_loader) 

def kfold_cv(
    dataset: torch.utils.data.Dataset,
    cfg: DictConfig,
    k: int = 5,
) -> float:
    accs = []
    for train_idx, test_idx in KFold(n_splits=k).split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        regularizer = hydra.utils.instantiate(cfg.regularizer)
        model: torch.nn.Module = hydra.utils.instantiate(cfg.model, regularizer=regularizer)
        regularizer.assign_embedding(model.embedding_layer.weight)
        
        acc = train_test(model, train_loader, test_loader, regularizer, cfg)
        accs.append(acc)
    return np.mean(accs)


def grid_search(
    dataset: torch.utils.data.Dataset, 
    cfg: DictConfig,
) -> Tuple[str, float]:
    for name in ['dropout_rate', 'weight']:
        if name in cfg.regularizer:
            param_name = name
            param_values = cfg.regularizer[name]
    
    if isinstance(param_values, (float, int)):
        return (param_name, param_values)

    opt_acc = 0.0
    cfg_new = cfg.copy()
    for param_value in param_values:
        cfg_new.regularizer[param_name] = param_value
        acc = kfold_cv(dataset, cfg_new)
        logger.info(f"{param_name}: {param_value}, acc: {acc}")
        if opt_acc < acc:
            opt_acc = acc
            opt_param_value = param_value
    return (param_name, opt_param_value)
