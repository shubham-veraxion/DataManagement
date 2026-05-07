import logging
import os
from datetime import datetime

_LOG_DIR = "storage/logs"
_LOG_FILE_PATH: str | None = None
_REGISTERED_LOGGERS: dict[str, logging.Logger] = {}


def _build_log_path() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    return os.path.join(_LOG_DIR, f"run_{timestamp}_{pid}.log")


def _ensure_log_path() -> str:
    global _LOG_FILE_PATH
    if _LOG_FILE_PATH is None:
        _LOG_FILE_PATH = _build_log_path()
    return _LOG_FILE_PATH


def _attach_file_handler(logger: logging.Logger, log_path: str) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def start_new_run() -> str:
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_path = _build_log_path()
    global _LOG_FILE_PATH
    _LOG_FILE_PATH = log_path

    for logger in _REGISTERED_LOGGERS.values():
        _attach_file_handler(logger, log_path)

    return log_path


def get_log_file_path() -> str:
    os.makedirs(_LOG_DIR, exist_ok=True)
    return _ensure_log_path()


def get_logger(name: str):
    os.makedirs(_LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    _REGISTERED_LOGGERS[name] = logger
    _attach_file_handler(logger, _ensure_log_path())

    return logger