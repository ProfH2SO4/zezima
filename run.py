from types import ModuleType
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import device

from zezima.utils.dataloader import LimitedDataset
from zezima.models.my_model import TransformerModel

import config


from zezima.training.train import train_model
from zezima.training.test import test_model, validate_model
from zezima import log


def get_device() -> device:
    """
    Determines the most suitable computing device available (CUDA-enabled GPU or CPU) and logs the selection.

    :return: The selected computing device, represented as a torch.device object.
    This will be CUDA if a compatible GPU is available, otherwise CPU.
    """
    target_device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log.info(f"CUDA is available. Using {target_device}.")
        print(f"CUDA is available. Using {target_device}.")
    else:
        log.info(f"CUDA is not available. Using {target_device}.")
        print(f"CUDA is not available. Using {target_device}.")
    return target_device


def load_config() -> ModuleType:
    """
    Loads the configuration from a local `config.py` file.
     If a `config.py` file exists in `/etc/zezima/`, it overrides parameters in the local `config.py`.

    :return: The configuration module loaded with settings from the configuration file.
    """
    app_config: ModuleType = config
    path: str = "/etc/zezima/config.py"

    if not os.path.isfile(path):
        return app_config
    try:
        with open(path, "rb") as rnf:
            exec(compile(rnf.read(), path, "exec"), app_config.__dict__)
    except OSError as e:
        print(f"File at {path} could not be loaded because of error: {e}")
        raise e from e
    return app_config


def parse_namespace(config_: ModuleType) -> dict[str, any]:
    """
    Parses and filters the attributes of a configuration module, excluding any built-in attributes.

    :param config_: The configuration module to be parsed.
    :return: A dictionary containing the filtered configuration parameters.
    """
    parsed: dict[str, any] = {}
    for key, value in config_.__dict__.items():
        if not key.startswith("__"):
            parsed[key] = value
    return parsed


def create_file_if_not_exists(path_to_file: str) -> None:
    """
    Checks if a file exists at the specified path, and if not, creates the file along with any necessary directories.

    :param path_to_file: The full path to the file that needs to be checked and potentially created.
    :return: None. The function's purpose is to ensure the file exists, not to return any value.
    """
    directory = os.path.dirname(path_to_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(path_to_file):
        with open(path_to_file, "w") as file:
            pass  # Create an empty file


def setup_model_data_loader(
    file: str, parsed_config: dict
) -> tuple[TransformerModel, DataLoader, nn.CrossEntropyLoss, torch.Tensor]:
    """
    Prepares and returns the components required for training a Transformer model including the model itself,
    a DataLoader for the dataset, the loss function, and an initial state matrix.

    :param file: The path to the dataset file to be used for training the model.
    :param parsed_config: A dictionary containing configuration parameters such as sequence length, model dimensions, batch size, etc.
    :return: A tuple containing the Transformer model, DataLoader for the dataset, the loss criterion (CrossEntropyLoss),
             and an initial state matrix used for training.
    """
    dataset = LimitedDataset(
        file,
        cpu_cores=parsed_config["NUM_CPU_CORES_DATASET"],
        bp_per_batch=parsed_config["SEQUENCE_LENGTH"],
        d_model=parsed_config["D_MODEL"],
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
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
        1,
        parsed_config["SEQUENCE_LENGTH"],
        parsed_config["D_MODEL"],
    )
    log.info(f"Number of workers: {data_loader.num_workers}")
    return model, data_loader, criterion, state_matrix


def prepare_model_and_state(
    model: TransformerModel,
    state_matrix: torch.Tensor,
    target_device: device,
    dtype=torch.float64,
) -> tuple[TransformerModel, torch.Tensor]:
    """
    Moves the model and state matrix to the specified device and sets the model's dtype to the specified type.

    Parameters:
    :param model: The Transformer model to be prepared.
    :param state_matrix: The state matrix to save memory between windows.
    :param target_device: The computing device (e.g., CPU or GPU) where the model and state matrix will be moved.
    :param dtype: The desired data type for the model's parameters and tensors.

    :return: model, state_matrix (after being moved to the target device.)
    """
    model = model.to(target_device)
    model = model.type(dtype)
    state_matrix = state_matrix.to(target_device).type(dtype)
    return model, state_matrix


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
    target_device: device = get_device()
    for file in train_files:
        model, data_loader, loss_function, state_matrix = setup_model_data_loader(
            file, parsed_config
        )

        model, state_matrix = prepare_model_and_state(
            model, state_matrix, target_device, dtype=torch.float64
        )
        if parsed_config["TRAIN_MODE"]:
            log.info(f"Training model on {file}")
            optimizer = optim.Adam(
                model.parameters(), lr=parsed_config["LEARNING_RATE"]
            )
            train_model(
                model,
                loss_function,
                optimizer,
                data_loader,
                state_matrix,
                parsed_config["NUM_EPOCHS"],
                parsed_config["MODEL_PATH"],
                target_device,
                parsed_config["CHECKPOINT_PATH"],
            )

        if parsed_config["VALIDATE_MODE"]:
            log.info(f"Validating model on {file}")
            model.load_state_dict(torch.load(parsed_config["MODEL_PATH"]))
            validate_model(model, loss_function, data_loader, state_matrix)
    if parsed_config["TEST_MODE"]:
        for file in test_files:
            log.info(f"Testing model on {file}")
            model, data_loader, loss_function, state_matrix = setup_model_data_loader(
                file, parsed_config
            )

            model.load_state_dict(torch.load(parsed_config["MODEL_PATH"]))
            model, state_matrix = prepare_model_and_state(
                model, state_matrix, target_device, dtype=torch.float64
            )
            test_model(model, loss_function, data_loader, state_matrix, target_device)

            log.info(f"Processed {file}")
    log.info("Done")
    print("Done")


if __name__ == "__main__":
    main()
