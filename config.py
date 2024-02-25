INPUT_TRAIN_DIRECTORY = "./data/fake_train_data"
INPUT_TEST_DIRECTORY = "./data/fake_test_data"
MODEL_PATH = "./saved_models/model_state.pth"
CHECKPOINT_PATH = "./checkpoint_models/fake_train_data.pth"


USE_CHECKPOINT = False
DEBUG_LEVEL = 2  #  Fow now only [1, 2], 2 => logs every iteration

TRAIN_MODE = True
VALIDATE_MODE = False
TEST_MODE = True
NUM_CPU_CORES_DATASET = 2

# Optimizer
LEARNING_RATE = 0.02

# Model
D_MODEL = 32
NHEAD = 16
NUM_ENCODER_LAYERS = 6
DIM_FEEDFORWARD = 128
SEQUENCE_LENGTH = 100
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
