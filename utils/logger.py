import logging
import logging.handlers
import logging_loki
from multiprocessing import Queue
import os
import sys
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

APPLICATON_NAME = "sc-clip"


class TabulateLevels(Enum):
    PRIMARY = "simple_grid"
    SECONDARY = "simple"


def setup_logger():
    # Fetch environment variables
    loki_url = os.getenv("LOKI_URL")
    loki_username = os.getenv("LOKI_USERNAME")
    loki_password = os.getenv("LOKI_PASSWORD")

    # Validate environment variables
    if not loki_url:
        raise ValueError("LOKI_URL environment variable is not set")
    if not loki_username:
        raise ValueError("LOKI_USERNAME environment variable is not set")
    if not loki_password:
        raise ValueError("LOKI_PASSWORD environment variable is not set")

    # Set up the logging queue and handler
    queue = Queue(-1)
    queue_handler = logging.handlers.QueueHandler(queue)

    # Set up the Loki handler
    handler_loki = logging_loki.LokiHandler(
        url=f"{loki_url}/loki/api/v1/push",
        tags={"application": APPLICATON_NAME},
        auth=(loki_username, loki_password),
        version="1",
    )

    # Set up the stdout handler for console logging
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stdout_handler.setFormatter(formatter)

    # Set up the listener to handle log entries from the queue
    listener = logging.handlers.QueueListener(queue, handler_loki, stdout_handler)

    # Start the listener
    listener.start()

    # Set up the root logger
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid double logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    return listener
