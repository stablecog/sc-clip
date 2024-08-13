import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from PIL import Image
from io import BytesIO
import requests
from urllib.parse import urlparse
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from flask import jsonify
import os
from dotenv import load_dotenv

load_dotenv()

TIMEOUT = os.getenv("TIMEOUT", 15)


@contextmanager
def time_log(after: str = "Completed", before: str | None = None):
    if before is not None:
        logging.info(before)
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        logging.info(f"{after}: {execution_time:.0f} ms")


def download_image(url, timeout=TIMEOUT):
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    return Image.open(BytesIO(response.content)).convert("RGB")


def download_images(urls, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, url) for url in urls]
        images = [future.result() for future in futures]
    return images


def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    logging.error(
                        f"🔴 Function {func.__name__} timed out after {seconds} seconds"
                    )
                    return jsonify({"error": "Operation timed out"}), 504

        return wrapper

    return decorator
