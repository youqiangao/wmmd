import logging
import hydra
from omegaconf import DictConfig
import torch
from sklearn.model_selection import train_test_split
from utils import set_seed, grid_search, train_test

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    dataset: torch.utils.data.Dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataset, test_dataset = train_test_split(dataset, test_size=cfg.test_size, random_state=cfg.seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    logger.info(f"Ruuning with dataset: \n{cfg.dataset}")
    
    # Grid search for hyperparameters using k-fold cross-validation
    if 'dropout_rate' in cfg.regularizer or 'weight' in cfg.regularizer:
        param_name, param_value = grid_search(train_dataset, cfg)
        cfg.regularizer[param_name] = param_value

    regularizer = hydra.utils.instantiate(cfg.regularizer)
    logger.info(f"Running with regularizer:\n{cfg.regularizer}")

    model: torch.nn.Module = hydra.utils.instantiate(cfg.model, regularizer=regularizer)
    regularizer.assign_embedding(model.embedding_layer.weight)
    logger.info(f"Running with model:\n{cfg.model}")

    train_test(model, train_loader, test_loader, regularizer, cfg, verbose=True)

if __name__ == "__main__":
    main()