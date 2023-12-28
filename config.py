INPUT_DIRECTORY = "./fake_data"
MODEL_PATH = "./saved_models/model_state.pth"

TRAIN_MODE = True
VALIDATE_MODE = True
TEST_MODE = True
BATCH_SIZE = 1
NUM_OF_WORKERS = 0

# Optimizer
LEARNING_RATE = 0.01
INPUT_SIZE = 10
D_MODEL = 10  # also represent a features
NHEAD = 1
NUM_ENCODER_LAYERS = 1
DIM_FEEDFORWARD = 1
SEQUENCE_LENGTH = 2

NUM_EPOCHS = 5


LOG_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "ZEZIMA - %(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "sys_logger6": {
            "level": "DEBUG",
            "class": "logging.handlers.SysLogHandler",
            "formatter": "default",
            "address": "/dev/log",
            "facility": "local6",
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",  # Use standard output
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": "./logs/default.txt",  # Specify the file path
        },
    },
    "loggers": {
        "default": {
            "level": "DEBUG",
            "handlers": ["sys_logger6", "console", "file"],
            "propagate": False,
        }
    },
    "disable_existing_loggers": False,
}
