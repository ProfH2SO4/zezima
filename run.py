from types import ModuleType
from os.path import isfile
import os, random

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
        if not key.startswith("__"):
            parsed[key] = value
    return parsed


def create_file_if_not_exists(path_to_file: str) -> None:
    directory = os.path.dirname(path_to_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(path_to_file):
        with open(path_to_file, "w") as file:
            pass  # Create an empty file


def setup_model_data_loader(file, parsed_config):
    dataset = LimitedDataset(
        file,
        bp_per_batch=parsed_config["SEQUENCE_LENGTH"],
        d_model=parsed_config["D_MODEL"],
    )
    data_loader = DataLoader(
        dataset,
        batch_size=parsed_config["BATCH_SIZE"],
        shuffle=False,
        num_workers=parsed_config["NUM_OF_WORKERS"],
    )
    model = TransformerModel(
        input_size=parsed_config["D_MODEL"],
        d_model=parsed_config["D_MODEL"],
        nhead=parsed_config["NHEAD"],
        num_encoder_layers=parsed_config["NUM_ENCODER_LAYERS"],
        dim_feedforward=parsed_config["DIM_FEEDFORWARD"],
        seq_length=parsed_config["SEQUENCE_LENGTH"],
    )
    criterion = nn.CrossEntropyLoss()
    state_matrix = torch.zeros(
        parsed_config["BATCH_SIZE"],
        parsed_config["SEQUENCE_LENGTH"],
        parsed_config["D_MODEL"],
    )
    return model, data_loader, criterion, state_matrix


def main() -> None:
    config_: ModuleType = load_config()
    parsed_config: dict[str, any] = parse_namespace(config_)

    print("============ Setting Up Logger ============")
    if config_.LOG_CONFIG["handlers"].get("file", None):
        file_path: str = config_.LOG_CONFIG["handlers"]["file"].get("filename")
        create_file_if_not_exists(file_path)
    log.set_up_logger(config_.LOG_CONFIG)

    train_directory: str = parsed_config["INPUT_TRAIN_DIRECTORY"]
    test_directory: str = parsed_config["INPUT_TEST_DIRECTORY"]

    train_files: list[str] = [
        os.path.join(train_directory, f)
        for f in os.listdir(train_directory)
        if f.endswith(".txt")
    ]
    test_files: list[str] = [
        os.path.join(test_directory, f)
        for f in os.listdir(test_directory)
        if f.endswith(".txt")
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log.info(f"CUDA is available. Using {device}.")
        print(f"CUDA is available. Using {device}.")
    else:
        log.info(f"CUDA is not available. Using {device}.")
        print(f"CUDA is not available. Using {device}.")

    for file in train_files:
        model, data_loader, criterion, state_matrix = setup_model_data_loader(
            file, parsed_config
        )
        model.to(device)
        state_matrix = state_matrix.to(device)
        model.double()
        if TRAIN_MODE:
            log.info(f"Training model on {file}")
            optimizer = optim.Adam(
                model.parameters(), lr=parsed_config["LEARNING_RATE"]
            )
            train_model(
                model,
                criterion,
                optimizer,
                data_loader,
                state_matrix,
                parsed_config["NUM_EPOCHS"],
                parsed_config["MODEL_PATH"],
                device,
            )

        if VALIDATE_MODE:
            log.info(f"Validating model on {file}")
            model.load_state_dict(torch.load(parsed_config["MODEL_PATH"]))
            validate_model(model, criterion, data_loader, state_matrix)
    if TEST_MODE:
        for file in test_files:
            log.info(f"Testing model on {file}")
            model, data_loader, criterion, state_matrix = setup_model_data_loader(
                file, parsed_config
            )
            model.load_state_dict(torch.load(parsed_config["MODEL_PATH"]))
            test_model(model, criterion, data_loader, state_matrix)

        log.info(f"Processed {file}")
    log.info("Done")
    print("Done")


if __name__ == "__main__":
    main()
