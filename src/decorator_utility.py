import functools
import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Decorator utility for safe execution of functions with error handling and logging.
def setup_logging(log_dir="logs", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"etl_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger.info("Logging initialized: %s", log_path)
    return log_path


# Decorator for safe execution of functions with error handling and logging.
def safe_execute(default=None, log_error=True, reraise=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            started_at = time.perf_counter()
            try:
                logger.info("Starting %s", func.__name__)
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.exception(f"Error in {func.__name__}: {e}")
                
                if reraise:
                    raise
                
                return default
            finally:
                elapsed_seconds = time.perf_counter() - started_at
                logger.info("Finished %s in %.3f seconds", func.__name__, elapsed_seconds)
        return wrapper
    return decorator