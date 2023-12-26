from types import ModuleType
from os.path import isfile
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from zezima.utils.dataloader import LimitedDataset
from zezima.models.my_model import TransformerModel

import config
from config import TRAIN_MODE, VALIDATE_MODE, TEST_MODE


from zezima.training.train import train_model
from zezima.training.test import test_model, validate_model
from zezima import log


def load_config() -> ModuleType:
    """
    Load local config.py.
    If exists config.py in /etc/zezima/ then overrides parameters in local config.py.
    @return: configuration file
    """
    app_config: ModuleType = config
    path: str = "/etc/zezima/config.py"

    if not isfile(path):
        return app_config
    try:
        with open(path, "rb") as rnf:
            exec(compile(rnf.read(), "config.py", "exec"), app_config.__dict__)
    except OSError as e:
        print(f"File at {path} could not be loaded because of error: {e}")
        raise e from e
    return app_config


def parse_namespace(config_: ModuleType) -> dict[str, any]:
    """
    Parse configuration file file to dict.
    @param config_: configuration file
    @return: parsed configuration file
    """
    parsed: dict[str, any] = {}
    for key, value in config_.__dict__.items():
        if not key.startswith('__'):
            parsed[key] = value
    return parsed


def main():
    config_: ModuleType = load_config()
    parsed_config: dict[str, any] = parse_namespace(config_)

    print("============ Setting Up Logger ============")
    log.set_up_logger(config_.LOG_CONFIG)

    directory = parsed_config["INPUT_DIRECTORY"]
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    for file in files:
        log.info(f"Starting {file}")
        dataset = LimitedDataset(file)
        data_loader = DataLoader(dataset, batch_size=1,
                                 shuffle=False, num_workers=parsed_config["NUM_OF_WORKERS"])

        model = TransformerModel(input_size=parsed_config["INPUT_SIZE"],
                                 d_model=parsed_config["D_MODEL"],
                                 nhead=parsed_config["NHEAD"],
                                 num_encoder_layers=parsed_config["NUM_ENCODER_LAYERS"],
                                 dim_feedforward=parsed_config["DIM_FEEDFORWARD"],
                                 seq_length=parsed_config["SEQUENCE_LENGTH"],
                                 )

        # Loss function and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=parsed_config["LEARNING_RATE"])
        state_matrix = torch.zeros(parsed_config["BATCH_SIZE"], parsed_config["SEQUENCE_LENGTH"], parsed_config["D_MODEL"])

        if TRAIN_MODE:
            log.debug("Training model")
            train_model(model, criterion, optimizer, data_loader, state_matrix, parsed_config["NUM_EPOCHS"])
        if VALIDATE_MODE:
            log.debug("Validate model")
            validate_model(model, criterion, data_loader, state_matrix)
        if TEST_MODE:
            log.debug("Testing model")
            test_model(model, criterion, data_loader, state_matrix)
        log.info(f"Ending {file}")
    log.info("Done")


if __name__ == "__main__":
    main()







