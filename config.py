TRAIN_MODE = True
VALIDATE_MODE = True
TEST_MODE = False
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
    },
    "loggers": {
        "default": {"level": "DEBUG", "handlers": ["sys_logger6"], "propagate": True}
    },
    "disable_existing_loggers": False,
}
